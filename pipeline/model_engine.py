"""
Module D — Model Engine (ML Inference Interface)
Stub implementation with deterministic mock output generator.

Designed for easy swap with TFLite, ONNX, or PyTorch runtime.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Protocol
import logging
import time

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Torch availability (graceful fallback if not installed)
# ---------------------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False



# ---------------------------------------------------------------------------
# Interface contracts
# ---------------------------------------------------------------------------

@dataclass
class ModelInput:
    """Input to the ML inference engine."""
    signal_tensor: np.ndarray       # float32 [12, 5000]
    lead_mask: np.ndarray           # binary  [12]
    sampling_rate: float            # Hz

    def to_torch(self):
        """
        Convert signal_tensor and lead_mask to PyTorch tensors.

        Returns:
            (signal_tensor, lead_mask) as torch.FloatTensors:
                signal_tensor: [1, 12, N] (batch dim added)
                lead_mask:     [1, 12]
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. "
                "Install with: pip install torch"
            )
        sig = torch.from_numpy(
            self.signal_tensor.astype(np.float32)
        ).unsqueeze(0)   # [12, N] → [1, 12, N]
        mask = torch.from_numpy(
            self.lead_mask.astype(np.float32)
        ).unsqueeze(0)   # [12]   → [1, 12]
        return sig, mask



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

        # Convert waveforms to torch tensors
        if TORCH_AVAILABLE:
            sig_tensor, mask_tensor = model_input.to_torch()
            log.info("  Torch tensor shape: %s, mask: %s, dtype: %s",
                     sig_tensor.shape, mask_tensor.shape, sig_tensor.dtype)
        else:
            log.warning("  PyTorch not available — using numpy arrays")

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
# TFLite model engine
# ---------------------------------------------------------------------------

class TFLiteModelEngine:
    """
    TFLite inference engine for Android deployment.
    Uses tflite_runtime if available, otherwise tensorflow.lite.
    """

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._interpreter = None
        self._input_details = None
        self._output_details = None

        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter
            except ImportError:
                log.error("Neither tflite_runtime nor tensorflow found. "
                          "Install: pip install tflite-runtime")
                raise

        self._interpreter = Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        log.info("TFLite engine loaded: %s", model_path)
        for inp in self._input_details:
            log.info("  Input:  %s %s %s", inp["name"], inp["shape"], inp["dtype"])
        for out in self._output_details:
            log.info("  Output: %s %s %s", out["name"], out["shape"], out["dtype"])

    def is_ready(self) -> bool:
        return self._interpreter is not None

    def predict(self, model_input: ModelInput) -> ModelOutput:
        t0 = time.perf_counter()

        # Set inputs
        sig = model_input.signal_tensor.astype(np.float32)
        if sig.ndim == 2:
            sig = sig[np.newaxis, ...]  # add batch dim
        mask = model_input.lead_mask.astype(np.float32)
        if mask.ndim == 1:
            mask = mask[np.newaxis, ...]

        self._interpreter.set_tensor(self._input_details[0]["index"], sig)
        if len(self._input_details) > 1:
            self._interpreter.set_tensor(self._input_details[1]["index"], mask)

        self._interpreter.invoke()

        # Parse outputs (order matches export script)
        outputs = [self._interpreter.get_tensor(d["index"])
                    for d in self._output_details]

        # Map outputs to ModelOutput
        # Order: probs_binary[B,2], probs_specific[B,K],
        #        uncertainty_bin[B], uncertainty_spec[B],
        #        signal_quality[B], lead_importance[B,12]
        probs_binary = outputs[0][0] if len(outputs) > 0 else np.array([0.5, 0.5])
        probs_specific = outputs[1][0] if len(outputs) > 1 else np.zeros(5)
        uncertainty_spec = float(outputs[3][0]) if len(outputs) > 3 else 0.5
        signal_quality = float(outputs[4][0]) if len(outputs) > 4 else 0.5
        lead_importance = outputs[5][0] if len(outputs) > 5 else None

        # Build class probabilities dict
        class_probs = {}
        for i, prob in enumerate(probs_specific):
            cls_name = (DIAGNOSIS_CLASSES[i]
                        if i < len(DIAGNOSIS_CLASSES)
                        else f"Class_{i}")
            class_probs[cls_name] = float(prob)

        # Determine binary prediction
        is_abnormal = probs_binary[1] > probs_binary[0]
        binary_pred = "abnormal" if is_abnormal else "normal"

        confidence = float(1.0 - uncertainty_spec)

        latency = (time.perf_counter() - t0) * 1000.0

        return ModelOutput(
            binary_prediction=binary_pred,
            class_probabilities=class_probs,
            attention_weights=(lead_importance.astype(np.float32)
                                if lead_importance is not None else None),
            suspicious_segments=[],
            confidence_score=np.clip(confidence, 0, 1),
            inference_latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# ONNX Runtime engine (recommended for Android)
# ---------------------------------------------------------------------------

class OnnxRuntimeEngine:
    """
    ONNX Runtime inference engine.
    Lightweight and well-supported on Android via onnxruntime-mobile.
    """

    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX backend. "
                "Install: pip install onnxruntime"
            )

        # Use CPU execution provider (NNAPI can be added for Android)
        providers = ["CPUExecutionProvider"]
        try:
            # On Android, prefer NNAPI for hardware acceleration
            available = ort.get_available_providers()
            if "NnapiExecutionProvider" in available:
                providers.insert(0, "NnapiExecutionProvider")
                log.info("  Using NNAPI hardware acceleration")
        except Exception:
            pass

        self._session = ort.InferenceSession(
            model_path, providers=providers
        )
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]

        log.info("ONNX Runtime engine loaded: %s", model_path)
        log.info("  Inputs:  %s", self._input_names)
        log.info("  Outputs: %s", self._output_names)

    def is_ready(self) -> bool:
        return self._session is not None

    def predict(self, model_input: ModelInput) -> ModelOutput:
        t0 = time.perf_counter()

        # Prepare inputs with batch dim
        sig = model_input.signal_tensor.astype(np.float32)
        if sig.ndim == 2:
            sig = sig[np.newaxis, ...]   # [12, N] -> [1, 12, N]
        mask = model_input.lead_mask.astype(np.float32)
        if mask.ndim == 1:
            mask = mask[np.newaxis, ...]  # [12] -> [1, 12]

        feed = {self._input_names[0]: sig}
        if len(self._input_names) > 1:
            feed[self._input_names[1]] = mask

        outputs = self._session.run(None, feed)

        # Parse outputs (order matches export script):
        # 0: probs_binary [B,2], 1: probs_specific [B,K]
        # 2: uncertainty_bin [B], 3: uncertainty_spec [B]
        # 4: signal_quality [B], 5: lead_importance [B,12]
        probs_binary = outputs[0][0] if len(outputs) > 0 else np.array([0.5, 0.5])
        probs_specific = outputs[1][0] if len(outputs) > 1 else np.zeros(5)
        uncertainty_spec = float(outputs[3][0]) if len(outputs) > 3 else 0.5
        signal_quality = float(outputs[4][0]) if len(outputs) > 4 else 0.5
        lead_importance = outputs[5][0] if len(outputs) > 5 else None

        # Build class probabilities dict
        class_probs = {}
        for i, prob in enumerate(probs_specific):
            cls_name = (DIAGNOSIS_CLASSES[i]
                        if i < len(DIAGNOSIS_CLASSES)
                        else f"Class_{i}")
            class_probs[cls_name] = float(prob)

        is_abnormal = probs_binary[1] > probs_binary[0]
        binary_pred = "abnormal" if is_abnormal else "normal"
        confidence = float(1.0 - uncertainty_spec)

        latency = (time.perf_counter() - t0) * 1000.0

        return ModelOutput(
            binary_prediction=binary_pred,
            class_probabilities=class_probs,
            attention_weights=(lead_importance.astype(np.float32)
                                if lead_importance is not None else None),
            suspicious_segments=[],
            confidence_score=float(np.clip(confidence, 0, 1)),
            inference_latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# TorchScript Mobile engine (for .ptl files)
# ---------------------------------------------------------------------------

class TorchMobileEngine:
    """
    PyTorch Mobile inference engine for .ptl (lite interpreter) files.
    """

    def __init__(self, model_path: str):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for TorchMobile backend. "
                               "pip install torch")
        self._model = torch.jit.load(model_path, map_location="cpu")
        self._model.eval()
        log.info("TorchMobile engine loaded: %s", model_path)

    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, model_input: ModelInput) -> ModelOutput:
        t0 = time.perf_counter()

        sig_t, mask_t = model_input.to_torch()

        with torch.no_grad():
            # TorchScript mobile returns tuple of tensors
            outputs = self._model(sig_t, mask_t)

        # Unpack: (probs_binary, probs_specific, uncertainty_spec,
        #          signal_quality, lead_importance)
        probs_binary = outputs[0][0].numpy()
        probs_specific = outputs[1][0].numpy()
        uncertainty_spec = float(outputs[2][0].item())
        signal_quality = float(outputs[3][0].item())
        lead_importance = outputs[4][0].numpy()

        class_probs = {}
        for i, prob in enumerate(probs_specific):
            cls_name = (DIAGNOSIS_CLASSES[i]
                        if i < len(DIAGNOSIS_CLASSES)
                        else f"Class_{i}")
            class_probs[cls_name] = float(prob)

        is_abnormal = probs_binary[1] > probs_binary[0]
        binary_pred = "abnormal" if is_abnormal else "normal"
        confidence = float(1.0 - uncertainty_spec)

        latency = (time.perf_counter() - t0) * 1000.0

        return ModelOutput(
            binary_prediction=binary_pred,
            class_probabilities=class_probs,
            attention_weights=lead_importance.astype(np.float32),
            suspicious_segments=[],
            confidence_score=np.clip(confidence, 0, 1),
            inference_latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_engine(
    backend: str = "mock",
    model_path: Optional[str] = None,
    seed: Optional[int] = None,
):
    """
    Factory function to create a model engine.

    Args:
        backend: "mock", "onnx", "tflite", "torchscript", or "auto".
        model_path: Path to model file (.onnx, .tflite, .ptl, .pt).
        seed: RNG seed for mock engine reproducibility.

    Returns:
        An engine implementing ModelEngineProtocol.
    """
    # Auto-detect backend from file extension
    if backend == "auto" and model_path:
        ext = model_path.rsplit(".", 1)[-1].lower()
        backend = {"onnx": "onnx", "tflite": "tflite",
                    "ptl": "torchscript", "pt": "torchscript"}.get(ext, "mock")
        log.info("Auto-detected backend: %s (from .%s)", backend, ext)

    if backend == "mock":
        return MockModelEngine(seed=seed)

    elif backend == "onnx":
        if not model_path:
            raise ValueError("model_path required for onnx backend")
        return OnnxRuntimeEngine(model_path)

    elif backend == "tflite":
        if not model_path:
            raise ValueError("model_path required for tflite backend")
        return TFLiteModelEngine(model_path)

    elif backend == "torchscript":
        if not model_path:
            raise ValueError("model_path required for torchscript backend")
        return TorchMobileEngine(model_path)

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
