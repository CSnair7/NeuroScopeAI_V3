"""
Module B — Signal Processor
Filtering, resampling, normalization, and pixel-to-voltage conversion.

Incorporates techniques from ECG-Image-Kit (Sameni et al., 2024):
  - Spatial filtering for trace extraction (image_to_sequence methods)
  - Proper ECG standard unit conversions (10mm/mV, 25mm/s)
  - Shadow removal via low-pass normalization
  - Combined-neighbors filtering for robust peak detection

Reference:
  ECG-Image-Kit: https://github.com/alphanumericslab/ecg-image-kit
"""

import numpy as np
from scipy import signal, ndimage
from dataclasses import dataclass
from typing import Optional
import logging
import cv2

from config import PIPELINE_SAMPLE_RATE_HZ

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ECG standard physical constants (from ECG-Image-Kit helper_functions.py)
# ---------------------------------------------------------------------------

# Standard ECG paper:  25 mm/s horizontal, 10 mm/mV vertical
ECG_PAPER_SPEED_MM_PER_S = 25.0    # mm per second
ECG_PAPER_GAIN_MM_PER_MV = 10.0    # mm per millivolt


def mm_to_volts(mm: float) -> float:
    """Convert vertical distance in mm to millivolts (10 mm = 1 mV)."""
    return float(mm / ECG_PAPER_GAIN_MM_PER_MV)


def mm_to_seconds(mm: float) -> float:
    """Convert horizontal distance in mm to seconds (25 mm = 1 s)."""
    return float(mm / ECG_PAPER_SPEED_MM_PER_S)


def px_to_mv(px: float, mm_per_px_y: float) -> float:
    """Convert pixel distance to millivolts using calibration."""
    return mm_to_volts(px * mm_per_px_y)


def px_to_seconds(px: float, mm_per_px_x: float) -> float:
    """Convert pixel distance to seconds using calibration."""
    return mm_to_seconds(px * mm_per_px_x)


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
    raw_pixel_row: np.ndarray = None  # centroid row positions within strip (px)


# ---------------------------------------------------------------------------
# Shadow removal (from ecg_gridest_spectral.m / ecg_gridest_margdist.m)
# ---------------------------------------------------------------------------

def remove_shadows(img_gray: np.ndarray, sigma_frac: float = 0.08) -> np.ndarray:
    """
    Remove uneven illumination / scanning shadows by dividing by a heavily
    blurred version of the image, then re-normalizing to [0, 1].

    Python port of the shadow removal step in ECG-Image-Kit's
    ecg_gridest_spectral.m (lines 170-183).

    Args:
        img_gray: Grayscale float image, values in [0, 1].
        sigma_frac: Blur sigma as a fraction of image diagonal.

    Returns:
        Shadow-free normalized grayscale image in [0, 1].
    """
    h, w = img_gray.shape
    sigma = sigma_frac * np.sqrt(h**2 + w**2)
    blurred = cv2.GaussianBlur(img_gray, (0, 0), sigma)
    blurred = np.clip(blurred, 1e-6, None)  # avoid division by zero
    normalized = img_gray / blurred
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-9)
    return normalized


# ---------------------------------------------------------------------------
# Image-to-sequence extraction (from image_to_sequence.m)
# ---------------------------------------------------------------------------

def image_to_sequence(
    img_strip: np.ndarray,
    mode: str = "bright-foreground",
    method: str = "combined_all_neighbors",
    window_len: int = 3,
) -> np.ndarray:
    """
    Extract a time-series from an ECG image strip using spatial filtering.

    Python port of ECG-Image-Kit's image_to_sequence.m.

    Instead of naively taking the topmost bright pixel per column, this
    applies spatial filters to suppress noise before finding the trace peak,
    producing a much cleaner signal.

    Args:
        img_strip: 2D grayscale image (float or uint8). Should be a single
                   lead strip.
        mode: 'bright-foreground' (trace is bright on dark) or
              'dark-foreground' (trace is dark on bright).
        method: Extraction method:
            'max_finder'     - raw argmax per column
            'moving_average' - box filter smoothing
            'hor_smoothing'  - horizontal-only smoothing
            'combined_all_neighbors' - min of horizontal, diagonal-down-right,
                                       diagonal-down-left filters (best)
        window_len: Filter kernel size.

    Returns:
        1D array of vertical positions (row indices), length = image width.
        Higher values = higher on the image = higher voltage.
    """
    # Ensure float
    if img_strip.dtype == np.uint8:
        img_f = img_strip.astype(np.float32) / 255.0
    else:
        img_f = img_strip.astype(np.float32)

    # Handle foreground mode
    if mode == "dark-foreground":
        img_f = 1.0 - img_f  # complement

    # Apply spatial filter
    if method == "max_finder":
        img_filtered = img_f

    elif method == "moving_average":
        kernel = np.ones((window_len, window_len), dtype=np.float32)
        kernel /= kernel.sum()
        img_filtered = cv2.filter2D(img_f, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    elif method == "hor_smoothing":
        kernel = np.ones((1, window_len), dtype=np.float32)
        kernel /= kernel.sum()
        img_filtered = cv2.filter2D(img_f, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    elif method == "combined_all_neighbors":
        # Three complementary filters - take the minimum response.
        # This suppresses isolated noise while preserving the continuous trace.
        h1 = np.array([[1, 1, 1]], dtype=np.float32) / 3.0
        z1 = cv2.filter2D(img_f, -1, h1, borderType=cv2.BORDER_REPLICATE)

        h2 = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]], dtype=np.float32) / 3.0
        z2 = cv2.filter2D(img_f, -1, h2, borderType=cv2.BORDER_REPLICATE)

        h3 = np.array([[0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]], dtype=np.float32) / 3.0
        z3 = cv2.filter2D(img_f, -1, h3, borderType=cv2.BORDER_REPLICATE)

        img_filtered = np.minimum(np.minimum(z1, z2), z3)
    else:
        img_filtered = img_f

    # Find the maximum pixel intensity in each column -> trace position
    row_indices = np.argmax(img_filtered, axis=0)

    # Convert to "amplitude" - invert so that higher on image = higher value
    img_height = img_filtered.shape[0]
    data = img_height - row_indices

    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Spectral grid calibration (from ecg_gridest_spectral.m)
# ---------------------------------------------------------------------------

def spectral_grid_estimation(
    img_gray: np.ndarray,
    num_patches: int = 50,
    num_seg: int = 5,
    min_grid_px: int = 3,
    min_prominence_db: float = 1.0,
) -> tuple:
    """
    Estimate grid line spacing using spectral analysis of random image patches.

    Python port of ECG-Image-Kit's ecg_gridest_spectral.m.

    Tiles the image into patches, computes the average power spectrum,
    and finds spectral peaks corresponding to grid line periodicities.

    Args:
        img_gray: Grayscale image (float, 0-1 or uint8).
        num_patches: Number of random patches to sample.
        num_seg: Number of segments in each direction for patch sizing.
        min_grid_px: Minimum grid size in pixels.
        min_prominence_db: Minimum peak prominence in dB.

    Returns:
        (grid_size_hor, grid_size_ver) — estimated grid spacings in pixels.
        Returns (0.0, 0.0) if estimation fails.
    """
    if img_gray.dtype == np.uint8:
        img_f = img_gray.astype(np.float32) / 255.0
    else:
        img_f = img_gray.astype(np.float32)

    h, w = img_f.shape

    # Shadow removal
    img_f = remove_shadows(img_f)

    seg_w = w // num_seg
    seg_h = h // num_seg

    if seg_w < 10 or seg_h < 10:
        return (0.0, 0.0)

    # Accumulate spectra from random patches
    spectra_stack = np.zeros((seg_h, seg_w), dtype=np.float64)

    rng = np.random.default_rng(42)
    count = 0
    for _ in range(num_patches):
        sx = rng.integers(0, w - seg_w)
        sy = rng.integers(0, h - seg_h)
        patch = img_f[sy:sy + seg_h, sx:sx + seg_w]

        # Apply Gaussian window
        win_y = np.hanning(seg_h).reshape(-1, 1)
        win_x = np.hanning(seg_w).reshape(1, -1)
        windowed = patch * win_y * win_x

        spectrum = np.abs(np.fft.fft2(windowed)) ** 2 / (seg_w * seg_h)
        spectra_stack += spectrum
        count += 1

    if count == 0:
        return (0.0, 0.0)

    spectra_avg = spectra_stack / count

    # Horizontal grid: average spectrum along columns
    spec_hor = 10 * np.log10(spectra_avg.mean(axis=1) + 1e-12)
    # Vertical grid: average spectrum along rows
    spec_ver = 10 * np.log10(spectra_avg.mean(axis=0) + 1e-12)

    def _find_grid_size(spec_1d):
        n = len(spec_1d)
        freqs = np.arange(n) / n
        # Only look at first half (up to Nyquist)
        half = n // 2
        spec_half = spec_1d[:half]
        freqs_half = freqs[:half]

        # Skip DC
        spec_half = spec_half[1:]
        freqs_half = freqs_half[1:]

        if len(spec_half) < 3:
            return 0.0

        # Find peaks
        peaks, properties = signal.find_peaks(
            spec_half,
            distance=min_grid_px,
            prominence=min_prominence_db,
        )

        if len(peaks) == 0:
            return 0.0

        # Sort by prominence
        proms = properties["prominences"]
        best = peaks[np.argmax(proms)]
        grid_freq = freqs_half[best]

        if grid_freq < 1e-6:
            return 0.0

        return 1.0 / grid_freq

    grid_hor = _find_grid_size(spec_hor)
    grid_ver = _find_grid_size(spec_ver)

    return (grid_hor, grid_ver)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_leads(
    mask: np.ndarray,
    mm_per_px_x: float,
    mm_per_px_y: float,
    lead_rows: Optional[list] = None,
    sample_rate_hz: float = PIPELINE_SAMPLE_RATE_HZ,
    original_image: Optional[np.ndarray] = None,
) -> list[LeadSignal]:
    """
    Convert a binary trace mask into per-lead voltage signals.

    Uses ECG-Image-Kit's image_to_sequence approach for more robust
    trace extraction, and standard ECG calibration for unit conversion.

    Args:
        mask: Binary mask (uint8, 0/255) of the ECG trace.
        mm_per_px_x: Horizontal spatial calibration.
        mm_per_px_y: Vertical spatial calibration.
        lead_rows: List of (y_start, y_end, lead_name) tuples.
                   If None, treats entire image as a single lead.
        sample_rate_hz: Target output sample rate.
        original_image: Optional original BGR/grayscale image for
                        combined_all_neighbors extraction.

    Returns:
        List of LeadSignal instances.
    """
    h, w = mask.shape
    if lead_rows is None:
        lead_rows = [(0, h, "Lead_I")]

    leads = []
    for y_start, y_end, name in lead_rows:
        row_mask = mask[y_start:y_end, :]

        # ---------------------------------------------------------------
        # Trace extraction — intensity-weighted centroid
        # ---------------------------------------------------------------
        # The trace mask may include grid lines (dim) along with the
        # actual ECG trace (bright). We use intensity^3 weighting so
        # bright trace pixels dominate the centroid, while dim grid
        # line pixels contribute negligibly.

        strip_h = y_end - y_start
        mask_f = row_mask.astype(np.float32) / 255.0

        # Build intensity weights — cube the intensity to strongly
        # favor the brightest pixels (the trace, not grid lines)
        if original_image is not None:
            if original_image.ndim == 3:
                img_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = original_image
            img_strip = img_gray[y_start:y_end, :].astype(np.float32) / 255.0

            # Per-column max to find trace intensity threshold
            col_max = (img_strip * mask_f).max(axis=0)
            # Threshold at 50% of each column's max — suppresses grid lines
            thresh = np.maximum(col_max * 0.5, 0.15).reshape(1, -1)
            bright_mask = (img_strip * mask_f) >= thresh

            # weights = intensity^3 * bright_mask
            weights = (img_strip ** 3) * bright_mask.astype(np.float32)
        else:
            weights = mask_f

        # Row index array: 0 = top of strip, strip_h-1 = bottom
        row_indices = np.arange(strip_h, dtype=np.float32).reshape(-1, 1)

        col_sum = weights.sum(axis=0)  # [W]
        col_has_trace = col_sum > 0

        # Weighted centroid per column (row position within strip)
        centroid_rows = np.full(w, strip_h / 2.0, dtype=np.float32)
        raw_amplitude = np.zeros(w, dtype=np.float32)
        valid = col_has_trace
        if valid.any():
            row_weighted = (weights * row_indices).sum(axis=0)
            centroid_rows[valid] = row_weighted[valid] / col_sum[valid]

            # Interpolate through gaps
            if not valid.all():
                x_valid = np.where(valid)[0]
                x_invalid = np.where(~valid)[0]
                centroid_rows[x_invalid] = np.interp(
                    x_invalid, x_valid, centroid_rows[x_valid]
                )

            # Light Gaussian smooth (sigma ~3px) to remove jitter
            from scipy.ndimage import gaussian_filter1d
            centroid_rows = gaussian_filter1d(centroid_rows, sigma=3.0)

            # Invert for voltage: lower row index = higher on image = higher voltage
            raw_amplitude = strip_h - centroid_rows

        # Continuity check
        col_has_trace = np.any(row_mask > 0, axis=0)
        continuity_pct = 100.0 * col_has_trace.sum() / w

        if continuity_pct < 2.0:
            log.warning("  %s: nearly empty (%.1f%% continuity) — skipping",
                        name, continuity_pct)
            continue

        # ---------------------------------------------------------------
        # Physical unit conversion (ECG-Image-Kit standard calibration)
        # ---------------------------------------------------------------
        # Baseline removal
        baseline_raw = ndimage.median_filter(
            raw_amplitude, size=max(3, w // 5), mode="nearest"
        )
        centered = raw_amplitude - baseline_raw

        # Pixel -> millivolts:  px * mm/px / (10 mm/mV)
        raw_voltage_mv = centered * mm_per_px_y / ECG_PAPER_GAIN_MM_PER_MV

        # Baseline drift in mV
        baseline_offset = baseline_raw - np.median(baseline_raw)
        baseline_drift_mv = baseline_offset * mm_per_px_y / ECG_PAPER_GAIN_MM_PER_MV

        # Time axis:  px * mm/px / (25 mm/s)
        duration_s = w * mm_per_px_x / ECG_PAPER_SPEED_MM_PER_S
        time_s = np.linspace(0, duration_s, w, dtype=np.float32)

        # Resample to target sample rate if needed
        actual_fs = w / duration_s if duration_s > 0 else sample_rate_hz
        if abs(actual_fs - sample_rate_hz) > 1.0:
            raw_voltage_mv = resample_signal(raw_voltage_mv, actual_fs, sample_rate_hz)
            baseline_drift_mv = resample_signal(baseline_drift_mv, actual_fs, sample_rate_hz)
            n_resampled = len(raw_voltage_mv)
            time_s = np.linspace(0, duration_s, n_resampled, dtype=np.float32)

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
            raw_pixel_row=centroid_rows,
        ))
        log.info("  %s — SNR: %.1f dB, continuity: %.1f%%, duration: %.2fs",
                 name, snr, continuity_pct, duration_s)

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
    """SNR = 20*log10(P_signal / P_noise) on QRS-gated windows."""
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
