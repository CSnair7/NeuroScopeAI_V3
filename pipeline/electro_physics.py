"""
Module C — Electro-Physics Reconstruction
Lead inference, reconstruction from Einthoven/Goldberger/Wilson laws,
lead mask generation, and signal tensor assembly.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

from config import NUM_LEADS, SIGNAL_TENSOR_LENGTH, PIPELINE_SAMPLE_RATE_HZ
from pipeline.signal_processor import LeadSignal, resample_signal, normalize_signal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard 12-lead ordering
# ---------------------------------------------------------------------------

LEAD_NAMES = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

LEAD_INDEX = {name: i for i, name in enumerate(LEAD_NAMES)}

# Aliases for flexible lead naming
LEAD_ALIASES = {
    "Lead_I": "I", "Lead_II": "II", "Lead_III": "III",
    "lead_I": "I", "lead_II": "II", "lead_III": "III",
    "lead_i": "I", "lead_ii": "II", "lead_iii": "III",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "v1": "V1", "v2": "V2", "v3": "V3",
    "v4": "V4", "v5": "V5", "v6": "V6",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReconstructionResult:
    """Output of lead reconstruction."""
    signal_tensor: np.ndarray       # float32 [12, SIGNAL_TENSOR_LENGTH]
    lead_mask: np.ndarray           # binary  [12]  (1=available, 0=missing/reconstructed)
    lead_names: list[str]           # ordered lead names
    rmse_reconstruction: float      # RMSE of reconstructed vs original (if verifiable)
    available_lead_count: int
    sampling_rate: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconstruct_leads(
    available_leads: list[LeadSignal],
    target_length: int = SIGNAL_TENSOR_LENGTH,
    target_fs: float = PIPELINE_SAMPLE_RATE_HZ,
) -> ReconstructionResult:
    """
    Given a set of extracted lead signals, reconstruct the full 12-lead
    tensor using electrocardiographic derivation rules.

    Derivation rules used:
        III  = II  - I         (Einthoven's law)
        aVR  = -(I + II) / 2  (Goldberger)
        aVL  = I - II/2       (Goldberger)
        aVF  = II - I/2       (Goldberger)

    Precordial leads (V1–V6) cannot be derived; they are zero-filled if missing.

    Args:
        available_leads: List of LeadSignal objects from signal processor.
        target_length: Number of samples per lead in the output tensor.
        target_fs: Target sampling rate for the output tensor.

    Returns:
        ReconstructionResult with [12, target_length] tensor and lead mask.
    """
    log.info("Electro-physics: reconstructing 12-lead tensor")

    # Map extracted leads to standard names
    lead_map: dict[str, np.ndarray] = {}
    for ls in available_leads:
        std_name = _standardize_name(ls.name)
        if std_name and std_name in LEAD_INDEX:
            # Resample to target length
            sig = ls.voltage_mv
            if len(sig) != target_length:
                sig = resample_signal(sig, len(sig), target_length)
            lead_map[std_name] = sig[:target_length]

    # Build tensor and mask
    tensor = np.zeros((NUM_LEADS, target_length), dtype=np.float32)
    mask = np.zeros(NUM_LEADS, dtype=np.float32)

    for name, sig in lead_map.items():
        idx = LEAD_INDEX[name]
        tensor[idx] = sig
        mask[idx] = 1.0

    # Derive missing limb leads
    rmse_terms = []

    # Einthoven: III = II - I
    if mask[LEAD_INDEX["I"]] and mask[LEAD_INDEX["II"]] and not mask[LEAD_INDEX["III"]]:
        derived = tensor[LEAD_INDEX["II"]] - tensor[LEAD_INDEX["I"]]
        tensor[LEAD_INDEX["III"]] = derived
        mask[LEAD_INDEX["III"]] = 0.5  # mark as derived
        log.info("  Derived III from Einthoven's law")

    # Verify Einthoven consistency if all 3 limb leads are available
    if all(mask[LEAD_INDEX[n]] >= 1.0 for n in ["I", "II", "III"]):
        expected_iii = tensor[LEAD_INDEX["II"]] - tensor[LEAD_INDEX["I"]]
        rmse = float(np.sqrt(np.mean(
            (tensor[LEAD_INDEX["III"]] - expected_iii) ** 2
        )))
        rmse_terms.append(rmse)
        log.info("  Einthoven RMSE: %.4f mV", rmse)

    # Goldberger augmented leads
    I = tensor[LEAD_INDEX["I"]]
    II = tensor[LEAD_INDEX["II"]]
    has_I_II = mask[LEAD_INDEX["I"]] > 0 and mask[LEAD_INDEX["II"]] > 0

    if has_I_II:
        if not mask[LEAD_INDEX["aVR"]]:
            tensor[LEAD_INDEX["aVR"]] = -(I + II) / 2.0
            mask[LEAD_INDEX["aVR"]] = 0.5
            log.info("  Derived aVR from Goldberger")

        if not mask[LEAD_INDEX["aVL"]]:
            tensor[LEAD_INDEX["aVL"]] = I - II / 2.0
            mask[LEAD_INDEX["aVL"]] = 0.5
            log.info("  Derived aVL from Goldberger")

        if not mask[LEAD_INDEX["aVF"]]:
            tensor[LEAD_INDEX["aVF"]] = II - I / 2.0
            mask[LEAD_INDEX["aVF"]] = 0.5
            log.info("  Derived aVF from Goldberger")

    # Precordial leads stay zero if not available
    for v_lead in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        if not mask[LEAD_INDEX[v_lead]]:
            log.info("  %s not available — zero-filled", v_lead)

    # Binarize mask for model input (available or derived ≥ 0.5 → 1)
    binary_mask = (mask >= 0.5).astype(np.float32)

    rmse_total = float(np.mean(rmse_terms)) if rmse_terms else 0.0
    available_count = int((mask >= 1.0).sum())

    log.info("  Tensor shape: %s, available: %d/12, derived: %d",
             tensor.shape, available_count,
             int((mask == 0.5).sum()))

    return ReconstructionResult(
        signal_tensor=tensor,
        lead_mask=binary_mask,
        lead_names=LEAD_NAMES,
        rmse_reconstruction=rmse_total,
        available_lead_count=available_count,
        sampling_rate=target_fs,
    )


def generate_lead_mask(leads: list[LeadSignal]) -> np.ndarray:
    """
    Quick utility: generate a binary [12] mask from available leads.
    """
    mask = np.zeros(NUM_LEADS, dtype=np.float32)
    for ls in leads:
        std_name = _standardize_name(ls.name)
        if std_name and std_name in LEAD_INDEX:
            mask[LEAD_INDEX[std_name]] = 1.0
    return mask


def build_signal_tensor(
    leads: list[LeadSignal],
    target_length: int = SIGNAL_TENSOR_LENGTH,
) -> np.ndarray:
    """
    Quick utility: build a [12, target_length] tensor from available leads.
    Missing leads are zero-filled.
    """
    tensor = np.zeros((NUM_LEADS, target_length), dtype=np.float32)
    for ls in leads:
        std_name = _standardize_name(ls.name)
        if std_name and std_name in LEAD_INDEX:
            sig = ls.voltage_mv
            if len(sig) >= target_length:
                tensor[LEAD_INDEX[std_name]] = sig[:target_length]
            else:
                tensor[LEAD_INDEX[std_name], :len(sig)] = sig
    return tensor


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _standardize_name(name: str) -> Optional[str]:
    """Map a lead name to its standard 12-lead name."""
    if name in LEAD_INDEX:
        return name
    return LEAD_ALIASES.get(name, None)
