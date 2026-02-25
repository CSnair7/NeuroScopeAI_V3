"""
Module D — Model Engine (ML Inference Interface)
Stub implementation with deterministic mock output generator.

Designed for easy swap with TFLite or ONNX runtime later.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Protocol
import logging
import time

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface contracts
# ---------------------------------------------------------------------------

@dataclass
class ModelInput:
    """Input to the ML inference engine."""
    signal_tensor: np.ndarray       # float32 [12, 5000]
    lead_mask: np.ndarray           # binary  [12]
    sampling_rate: float            # Hz


@dataclass
class ModelOutput:
    """Output from the ML inference engine."""
    binary_prediction: str                          # "normal" | "abnormal"
    class_probabilities: dict[str, float]           # class → probability
    attention_weights: Optional[np.ndarray]         # float [12] or None
    suspicious_segments: list[dict]                 # [{"start": int, "end": int}, ...]
    confidence_score: float                         # 0–1
    inference_latency_ms: float = 0.0


class ModelEngineProtocol(Protocol):
    """Protocol that any model engine (mock, TFLite, ONNX) must implement."""

    def predict(self, model_input: ModelInput) -> ModelOutput:
        ...

    def is_ready(self) -> bool:
        ...


# ---------------------------------------------------------------------------
# Diagnosis class labels
# ---------------------------------------------------------------------------

DIAGNOSIS_CLASSES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Supraventricular Tachycardia",
    "Ventricular Tachycardia",
    "Premature Ventricular Complex",
    "Left Bundle Branch Block",
    "Right Bundle Branch Block",
    "ST Elevation",
    "ST Depression",
]


# ---------------------------------------------------------------------------
# Mock model engine
# ---------------------------------------------------------------------------

class MockModelEngine:
    """
    Deterministic mock ML engine for UI testing.

    Behaviour:
        - Uses signal statistics to generate plausible-looking mock output.
        - ~70% normal, ~30% abnormal (controlled by signal energy).
        - Produces fake suspicious segment ranges for abnormal predictions.
        - Seed can be fixed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, model_input: ModelInput) -> ModelOutput:
        t0 = time.perf_counter()

        signal_energy = float(np.mean(model_input.signal_tensor ** 2))
        n_available = int(model_input.lead_mask.sum())

        # Decide normal vs abnormal based on a pseudo-random draw
        # weighted by signal characteristics
        abnormal_prob = self._rng.uniform(0.0, 1.0)
        is_abnormal = abnormal_prob > 0.7

        if is_abnormal:
            output = self._generate_abnormal(model_input, signal_energy)
        else:
            output = self._generate_normal(signal_energy)

        output.inference_latency_ms = (time.perf_counter() - t0) * 1000.0
        return output

    def _generate_normal(self, energy: float) -> ModelOutput:
        """Generate a normal sinus rhythm prediction."""
        confidence = float(np.clip(0.90 + self._rng.uniform(0, 0.08), 0.88, 0.99))
        nsr_prob = float(np.clip(0.92 + self._rng.uniform(0, 0.06), 0.90, 0.99))

        probs = {DIAGNOSIS_CLASSES[0]: nsr_prob}
        remaining = 1.0 - nsr_prob
        for cls in DIAGNOSIS_CLASSES[1:]:
            p = remaining * self._rng.uniform(0.001, 0.1)
            probs[cls] = round(float(p), 4)
            remaining -= p
            if remaining <= 0:
                break

        return ModelOutput(
            binary_prediction="normal",
            class_probabilities=probs,
            attention_weights=self._rng.uniform(0.02, 0.15, size=12).astype(np.float32),
            suspicious_segments=[],
            confidence_score=confidence,
        )

    def _generate_abnormal(
        self, model_input: ModelInput, energy: float
    ) -> ModelOutput:
        """Generate an abnormal prediction with suspicious segments."""
        # Pick top abnormal class
        abnormal_classes = DIAGNOSIS_CLASSES[1:]
        top_idx = self._rng.integers(0, len(abnormal_classes))
        top_class = abnormal_classes[top_idx]
        top_prob = float(self._rng.uniform(0.65, 0.92))

        # Second class
        second_idx = (top_idx + self._rng.integers(1, len(abnormal_classes))) % len(abnormal_classes)
        second_class = abnormal_classes[second_idx]
        second_prob = float(self._rng.uniform(0.05, 0.20))

        # NSR gets what's left
        nsr_prob = max(0.01, 1.0 - top_prob - second_prob)

        probs = {
            top_class: round(top_prob, 4),
            second_class: round(second_prob, 4),
            DIAGNOSIS_CLASSES[0]: round(nsr_prob, 4),
        }

        # Confidence
        confidence = float(np.clip(top_prob - 0.05 + self._rng.uniform(0, 0.1),
                                   0.5, 0.98))

        # Suspicious segments (1–3 regions)
        n_segments = self._rng.integers(1, 4)
        sig_len = model_input.signal_tensor.shape[1]
        segments = []
        for _ in range(n_segments):
            start = int(self._rng.integers(0, max(1, sig_len - 500)))
            end = int(min(start + self._rng.integers(100, 500), sig_len))
            segments.append({"start": start, "end": end})

        # Attention weights — simulate model looking at specific leads more
        attn = self._rng.uniform(0.02, 0.12, size=12).astype(np.float32)
        # Boost leads I, II, V1 for abnormal
        attn[0] *= 3.0
        attn[1] *= 2.5
        attn[6] *= 2.0
        attn /= attn.sum()

        return ModelOutput(
            binary_prediction="abnormal",
            class_probabilities=probs,
            attention_weights=attn,
            suspicious_segments=segments,
            confidence_score=confidence,
        )


# ---------------------------------------------------------------------------
# Factory function for future TFLite swap
# ---------------------------------------------------------------------------

def create_engine(
    backend: str = "mock",
    model_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> MockModelEngine:
    """
    Factory function to create a model engine.

    Args:
        backend: "mock" (default) or "tflite" (future).
        model_path: Path to TFLite/ONNX model file (future).
        seed: RNG seed for mock engine reproducibility.

    Returns:
        An engine implementing ModelEngineProtocol.
    """
    if backend == "mock":
        return MockModelEngine(seed=seed)
    elif backend == "tflite":
        # TODO: Implement TFLiteModelEngine
        raise NotImplementedError(
            "TFLite backend not yet implemented. "
            "Set backend='mock' for development testing."
        )
    elif backend == "onnx":
        # TODO: Implement ONNXModelEngine
        raise NotImplementedError(
            "ONNX backend not yet implemented."
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Helper: generate explanation text for suspicious segments
# ---------------------------------------------------------------------------

SEGMENT_EXPLANATIONS = [
    "Irregular R-R interval detected",
    "Wide QRS complex observed",
    "ST segment deviation noted",
    "Abnormal P-wave morphology",
    "Prolonged PR interval",
    "T-wave inversion detected",
    "Premature beat detected",
]


def get_segment_explanation(segment_index: int, prediction: str) -> str:
    """
    Return a human-readable explanation for a suspicious segment.
    In production, this would be driven by the model's attention map.
    """
    if prediction == "Atrial Fibrillation":
        return "Irregular R-R interval detected"
    elif prediction in ("Left Bundle Branch Block", "Right Bundle Branch Block"):
        return "Wide QRS complex observed"
    elif prediction in ("ST Elevation", "ST Depression"):
        return "ST segment deviation noted"
    else:
        return SEGMENT_EXPLANATIONS[segment_index % len(SEGMENT_EXPLANATIONS)]
