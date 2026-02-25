"""
Module B — Signal Processor
Filtering, resampling, normalization, and pixel-to-voltage conversion.

Refactored from extraction_pipeline.py (Steps 4–5) with added DSP utilities.
"""

import numpy as np
from scipy import signal, ndimage
from dataclasses import dataclass
from typing import Optional
import logging

from config import PIPELINE_SAMPLE_RATE_HZ

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LeadSignal:
    """Processed voltage signal for a single ECG lead."""
    name: str
    voltage_mv: np.ndarray          # millivolt waveform
    time_s: np.ndarray              # time axis in seconds
    snr_db: float                   # signal-to-noise ratio
    continuity_pct: float           # percent of non-gap pixels
    baseline_drift_mv: np.ndarray   # baseline wander component
    raw_voltage_mv: np.ndarray      # before filtering


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_leads(
    mask: np.ndarray,
    mm_per_px_x: float,
    mm_per_px_y: float,
    lead_rows: Optional[list] = None,
    sample_rate_hz: float = PIPELINE_SAMPLE_RATE_HZ,
) -> list[LeadSignal]:
    """
    Convert a binary trace mask into per-lead voltage signals.

    Args:
        mask: Binary mask (uint8, 0/255) of the ECG trace.
        mm_per_px_x: Horizontal spatial calibration.
        mm_per_px_y: Vertical spatial calibration.
        lead_rows: List of (y_start, y_end, lead_name) tuples.
                   If None, treats entire image as a single lead.
        sample_rate_hz: Target output sample rate.

    Returns:
        List of LeadSignal instances.
    """
    h, w = mask.shape
    if lead_rows is None:
        lead_rows = [(0, h, "Lead_I")]

    leads = []
    for y_start, y_end, name in lead_rows:
        row_mask = mask[y_start:y_end, :]

        # Extract topmost trace pixel per column (lower y = higher voltage)
        voltage_px = np.full(w, np.nan)
        for col in range(w):
            col_data = row_mask[:, col]
            trace_rows = np.where(col_data > 0)[0]
            if len(trace_rows):
                voltage_px[col] = float(trace_rows[0])

        # Continuity check
        nan_mask = np.isnan(voltage_px)
        continuity_pct = 100.0 * (1 - nan_mask.sum() / w)

        if nan_mask.all():
            log.warning("  %s: entirely missing — skipping", name)
            continue

        # Interpolate gaps
        idx = np.arange(w)
        voltage_px = np.interp(idx, idx[~nan_mask], voltage_px[~nan_mask])

        # Rolling median baseline (200ms window)
        window = max(3, int(0.2 * sample_rate_hz))
        baseline_px = ndimage.median_filter(voltage_px, size=window, mode="nearest")

        # Convert pixel position → millivolts
        # Negate: image y goes down, but voltage goes up
        raw_voltage_mv = -(voltage_px - baseline_px) * mm_per_px_y * 0.1
        baseline_drift_mv = (
            -(baseline_px - np.median(baseline_px)) * mm_per_px_y * 0.1
        )
        time_s = np.arange(w) / sample_rate_hz

        # Apply bandpass filter
        filtered_mv = bandpass_filter(raw_voltage_mv, sample_rate_hz)

        # Compute SNR
        snr = _compute_snr(filtered_mv)

        leads.append(LeadSignal(
            name=name,
            voltage_mv=filtered_mv,
            time_s=time_s,
            snr_db=snr,
            continuity_pct=continuity_pct,
            baseline_drift_mv=baseline_drift_mv,
            raw_voltage_mv=raw_voltage_mv,
        ))
        log.info("  %s — SNR: %.1f dB, continuity: %.1f%%",
                 name, snr, continuity_pct)

    return leads


# ---------------------------------------------------------------------------
# DSP utilities
# ---------------------------------------------------------------------------

def bandpass_filter(
    sig: np.ndarray,
    fs: float = PIPELINE_SAMPLE_RATE_HZ,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.

    Args:
        sig: Input signal (1D).
        fs: Sampling frequency in Hz.
        low: Lower cutoff in Hz.
        high: Upper cutoff in Hz.
        order: Filter order.

    Returns:
        Filtered signal.
    """
    nyq = fs / 2.0
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)

    if low_n >= high_n:
        return sig

    try:
        b, a = signal.butter(order, [low_n, high_n], btype="band")
        return signal.filtfilt(b, a, sig, padlen=min(3 * order, len(sig) - 1))
    except Exception:
        return sig


def resample_signal(
    sig: np.ndarray,
    original_fs: float,
    target_fs: float,
) -> np.ndarray:
    """
    Resample a 1D signal to a target sampling rate using polyphase resampling.
    """
    if abs(original_fs - target_fs) < 1e-6:
        return sig

    num_samples = int(len(sig) * target_fs / original_fs)
    return signal.resample(sig, num_samples)


def normalize_signal(sig: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization."""
    std = np.std(sig)
    if std < 1e-9:
        return sig - np.mean(sig)
    return (sig - np.mean(sig)) / std


def notch_filter(
    sig: np.ndarray,
    fs: float = PIPELINE_SAMPLE_RATE_HZ,
    freq: float = 50.0,
    Q: float = 30.0,
) -> np.ndarray:
    """Remove powerline interference (50/60 Hz)."""
    nyq = fs / 2.0
    if freq >= nyq:
        return sig
    b, a = signal.iirnotch(freq / nyq, Q)
    return signal.filtfilt(b, a, sig)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_snr(voltage_mv: np.ndarray) -> float:
    """SNR = 20·log10(P_signal / P_noise) on QRS-gated windows."""
    abs_v = np.abs(voltage_mv)
    if abs_v.max() < 1e-9:
        return 0.0

    threshold = 0.3 * abs_v.max()
    sig_mask = abs_v > threshold
    noise_mask = ~sig_mask

    p_sig = float(np.mean(voltage_mv[sig_mask] ** 2)) if sig_mask.any() else 1e-9
    p_noise = float(np.mean(voltage_mv[noise_mask] ** 2)) if noise_mask.any() else 1e-12

    if p_noise <= 0:
        return 99.0
    return float(20 * np.log10(np.sqrt(p_sig) / (np.sqrt(p_noise) + 1e-12)))
