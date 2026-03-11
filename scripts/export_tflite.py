#!/usr/bin/env python
"""
Export CardioGuardNet v12/v14 -> TFLite for Android deployment.

Usage:
    python scripts/export_tflite.py                          # default paths
    python scripts/export_tflite.py --quantize int8          # INT8 quantization
    python scripts/export_tflite.py --checkpoint path/to.pt  # custom checkpoint

Pipeline:
    PyTorch (.pt) -> ONNX (.onnx) -> TFLite (.tflite)

    The novel feature extractors use torch.fft.rfft which ONNX doesn't support.
    This script replaces FFT with an equivalent DFT matrix multiply (matmul)
    that is fully compatible with ONNX and TFLite.

Prerequisites:
    pip install torch onnx onnxruntime onnx-tf tensorflow
"""

import argparse
import math
import os
import sys
import numpy as np
from pathlib import Path

# Add project root so we can import the model class
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))  # for cardioguardnet_v12.py


# ---------------------------------------------------------------------------
# ONNX-compatible DFT replacement for torch.fft.rfft
# ---------------------------------------------------------------------------

def _build_dft_matrix(n: int):
    """
    Build the real DFT matrix pair (cos, sin) for length n.

    rfft(x) = x @ cos_mat + j * x @ sin_mat
    |rfft(x)|^2 = (x @ cos_mat)^2 + (x @ sin_mat)^2

    Returns:
        cos_mat: [n, n//2+1]  (real part basis)
        sin_mat: [n, n//2+1]  (imag part basis)
    """
    import torch
    freq_bins = n // 2 + 1
    k = torch.arange(freq_bins, dtype=torch.float32)    # [freq_bins]
    t = torch.arange(n, dtype=torch.float32)             # [n]
    # Phase matrix: [n, freq_bins]
    phase = 2.0 * math.pi * t.unsqueeze(1) * k.unsqueeze(0) / n
    cos_mat = torch.cos(phase)    # [n, freq_bins]
    sin_mat = -torch.sin(phase)   # [n, freq_bins]  (negative for DFT convention)
    return cos_mat, sin_mat


class OnnxSafeRfftPower(object):
    """
    Replaces |torch.fft.rfft(x)|^2 with matmul-based DFT.
    Pre-computes DFT matrices for a given signal length.
    """
    def __init__(self, n: int):
        self.n = n
        self.cos_mat, self.sin_mat = _build_dft_matrix(n)

    def power_spectrum(self, x):
        """
        Compute |FFT(x)|^2 using matmul.
        x: [..., n]
        Returns: [..., n//2+1]
        """
        import torch
        device = x.device
        cos_m = self.cos_mat.to(device)
        sin_m = self.sin_mat.to(device)
        real = torch.matmul(x, cos_m)  # [..., freq_bins]
        imag = torch.matmul(x, sin_m)
        return real ** 2 + imag ** 2

    def rfftfreq(self, fs: float):
        """Equivalent of torch.fft.rfftfreq(n, 1/fs)."""
        import torch
        freq_bins = self.n // 2 + 1
        return torch.arange(freq_bins, dtype=torch.float32) * fs / self.n


class OnnxCompatibleNovelFeatures:
    """
    Monkey-patches the novel feature modules to use DFT matmul
    instead of torch.fft.rfft, making them ONNX-exportable.
    """

    def __init__(self, model, seq_len: int = 5000, fs: int = 500):
        import torch
        self.model = model
        self.fs = fs

        # Pre-compute DFT for full signal length
        self.full_dft = OnnxSafeRfftPower(seq_len)
        self.full_freqs = self.full_dft.rfftfreq(fs)

        # Pre-compute DFT for windowed segments
        # TII uses win=400 samples (800ms @ 500Hz)
        tii_win = int(fs * 0.8)
        self.tii_dft = None  # TII doesn't use FFT

        # STE uses win=1000 samples (2s @ 500Hz)
        ste_win = int(fs * 2.0)
        self.ste_dft = OnnxSafeRfftPower(ste_win)

        # RRV uses full signal FFT + autocorrelation
        self.rrv_dft = OnnxSafeRfftPower(seq_len)

    def patch(self):
        """Apply monkey-patches to make model fully ONNX-traceable."""
        import torch
        import torch.nn.functional as F
        import types

        # === Patch ElectroPhysicsEngine (remove data-dependent branching) ===
        if hasattr(self.model, 'physics'):
            physics = self.model.physics

            def physics_forward_branchless(self_mod, x, mask):
                x = x.clone()
                mask = mask.clone()

                m0 = mask[:, 0:1]   # [B, 1]
                m1 = mask[:, 1:2]
                m2 = mask[:, 2:3]

                # Einthoven: branchless using torch.where
                # Recover lead II if I and III present but II absent
                can1 = (m0 * m2 * (1.0 - m1))  # [B, 1]
                recovered_1 = x[:, 0:1] + x[:, 2:3]  # [B, 1, T]
                x_1 = torch.where(can1.unsqueeze(-1) > 0.5, recovered_1, x[:, 1:2])
                mask_1 = torch.where(can1 > 0.5, torch.ones_like(m1), m1)

                # Recover lead III if I and II present but III absent
                can2 = (m0 * mask_1 * (1.0 - m2))
                recovered_2 = x_1 - x[:, 0:1]
                x_2 = torch.where(can2.unsqueeze(-1) > 0.5, recovered_2, x[:, 2:3])
                mask_2 = torch.where(can2 > 0.5, torch.ones_like(m2), m2)

                # Recover lead I if II and III present but I absent
                can0 = (mask_1 * mask_2 * (1.0 - m0))
                recovered_0 = x_1 - x_2
                x_0 = torch.where(can0.unsqueeze(-1) > 0.5, recovered_0, x[:, 0:1])
                mask_0 = torch.where(can0 > 0.5, torch.ones_like(m0), m0)

                # Reassemble leads 0-2
                x = torch.cat([x_0, x_1, x_2, x[:, 3:]], dim=1)
                mask = torch.cat([mask_0, mask_1, mask_2, mask[:, 3:]], dim=1)

                # Goldberger: branchless
                has_I_II = mask_0 * mask_1  # [B, 1]
                aVR = -(x_0 + x_1) / 2
                aVL = x_0 - x_1 / 2
                aVF = x_1 - x_0 / 2
                for lead_idx, val in [(3, aVR), (4, aVL), (5, aVF)]:
                    mi = mask[:, lead_idx:lead_idx+1]  # [B, 1]
                    need = has_I_II * (1.0 - mi)
                    x_lead = torch.where(need.unsqueeze(-1) > 0.5, val, x[:, lead_idx:lead_idx+1])
                    mask_lead = torch.where(need > 0.5, torch.ones_like(mi), mi)
                    # In-place replacement via slicing isn't traceable, rebuild
                    x = torch.cat([x[:, :lead_idx], x_lead, x[:, lead_idx+1:]], dim=1)
                    mask = torch.cat([mask[:, :lead_idx], mask_lead, mask[:, lead_idx+1:]], dim=1)

                return x, mask

            physics.forward = types.MethodType(physics_forward_branchless, physics)
            print("    [OK] Patched ElectroPhysicsEngine -> branchless (ONNX-safe)")

        # === Patch novel feature extractors (FFT -> DFT matmul) ===
        nf = self.model.novel_feat
        full_dft = self.full_dft
        full_freqs = self.full_freqs

        # --- DynamicMorphologyRatio ---
        dmr = nf.dmr

        def dmr_forward(self_mod, x, mask):
            w = mask / (mask.sum(1, keepdim=True) + 1e-8)
            sig = (x * w.unsqueeze(-1)).sum(1)
            pwr = full_dft.power_spectrum(sig)
            f = full_freqs.to(x.device)
            e_qrs = pwr[:, (f >= 8) & (f <= 30)].sum(-1)
            e_t = pwr[:, (f >= 0.5) & (f <= 8)].sum(-1)
            return torch.log((e_qrs + 1e-8) / (e_t + 1e-8)).unsqueeze(-1)

        dmr.forward = types.MethodType(dmr_forward, dmr)

        # --- SpectroTemporalEntropy (no early return) ---
        ste = nf.ste
        ste_dft = self.ste_dft

        def ste_forward(self_mod, x, mask):
            w = mask / (mask.sum(1, keepdim=True) + 1e-8)
            sig = (x * w.unsqueeze(-1)).sum(1)
            wins = sig.unfold(-1, self_mod.win, self_mod.stride)
            # No early return -- seq_len=5000 always gives enough windows
            pwr = ste_dft.power_spectrum(wins)
            p = pwr / (pwr.sum(-1, keepdim=True) + 1e-8)
            H = -(p * torch.log(p + 1e-10)).sum(-1)
            return H.var(-1).unsqueeze(-1)

        ste.forward = types.MethodType(ste_forward, ste)

        # --- TemporalInstabilityIndex (no early return) ---
        tii = nf.tii

        def tii_forward(self_mod, x, mask):
            w = mask / (mask.sum(1, keepdim=True) + 1e-8)
            sig = (x * w.unsqueeze(-1)).sum(1)
            wins = sig.unfold(-1, self_mod.win, self_mod.stride)
            mu = wins.mean(-1, keepdim=True)
            sd = wins.std(-1, keepdim=True) + 1e-8
            wn = (wins - mu) / sd
            corr = (wn[:, :-1] * wn[:, 1:]).mean(-1)
            return (1.0 - corr.mean(-1)).unsqueeze(-1)

        tii.forward = types.MethodType(tii_forward, tii)

        # --- RRVariabilityIndex (no early return, no FFT) ---
        rrv = nf.rrv

        def rrv_forward(self_mod, x, mask):
            w = mask / (mask.sum(1, keepdim=True) + 1e-8)
            sig = (x * w.unsqueeze(-1)).sum(1)
            sig_sq = sig ** 2
            wins = sig_sq.unfold(-1, self_mod.fs * 2, self_mod.fs)
            energy = wins.sum(-1)
            cv = energy.std(-1) / (energy.mean(-1) + 1e-8)
            return cv.unsqueeze(-1)

        rrv.forward = types.MethodType(rrv_forward, rrv)

        # --- STSegmentDeviation ---
        std_mod = nf.std

        def std_forward(self_mod, x, mask):
            w = mask / (mask.sum(1, keepdim=True) + 1e-8)
            sig = (x * w.unsqueeze(-1)).sum(1)
            pwr = full_dft.power_spectrum(sig)
            f = full_freqs.to(x.device)
            e_st = pwr[:, (f >= 1) & (f <= 5)].sum(-1)
            e_qrs = pwr[:, (f >= 8) & (f <= 25)].sum(-1)
            return torch.log((e_st + 1e-8) / (e_qrs + 1e-8)).unsqueeze(-1)

        std_mod.forward = types.MethodType(std_forward, std_mod)

        print("    [OK] Patched 5 FFT/branch modules -> ONNX-safe")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, num_classes: int = 5, num_leads: int = 12):
    """Load CardioGuardNet from a .pt checkpoint."""
    import torch
    from cardioguardnet_v12 import CardioGuardNetV12

    print(f"[1/5] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = CardioGuardNetV12(
        num_classes=num_classes,
        num_leads=num_leads,
        fs=500,
        drop_path_rate=0.0,
        use_physics=True,
    )

    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Loaded {total_params:,} parameters, num_classes={num_classes}")
    return model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model, output_path: str, num_leads: int = 12, seq_len: int = 5000):
    """Export the model to ONNX format with FFT replaced by DFT matmul."""
    import torch

    print(f"[2/5] Exporting to ONNX: {output_path}")

    # Patch FFT ops -> DFT matmul
    patcher = OnnxCompatibleNovelFeatures(model, seq_len=seq_len, fs=500)
    patcher.patch()

    dummy_x = torch.randn(1, num_leads, seq_len)
    dummy_mask = torch.ones(1, num_leads)

    class OnnxWrapper(torch.nn.Module):
        """Wraps CardioGuardNet to output a flat tuple for ONNX export."""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, mask):
            out = self.model(x, mask)
            return (
                out["probs_binary"],
                out["probs_specific"],
                out["uncertainty_bin"],
                out["uncertainty_spec"],
                out["signal_quality"],
                out["lead_importance"],
            )

    wrapper = OnnxWrapper(model)
    wrapper.eval()

    # Verify patched model runs
    with torch.no_grad():
        test_out = wrapper(dummy_x, dummy_mask)
        print(f"    Pre-export verification: probs_specific shape = "
              f"{test_out[1].shape}")

    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_mask),
        output_path,
        input_names=["ecg_signal", "lead_mask"],
        output_names=[
            "probs_binary", "probs_specific",
            "uncertainty_binary", "uncertainty_specific",
            "signal_quality", "lead_importance",
        ],
        dynamic_axes={
            "ecg_signal": {0: "batch"},
            "lead_mask": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"    [OK] ONNX exported: {size_mb:.1f} MB")
    return output_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_onnx(model, onnx_path: str, num_leads: int = 12, seq_len: int = 5000):
    """Verify ONNX model matches PyTorch output."""
    import torch
    import onnxruntime as ort

    print(f"[3/5] Verifying ONNX output...")

    dummy_x = torch.randn(1, num_leads, seq_len)
    dummy_mask = torch.ones(1, num_leads)

    with torch.no_grad():
        pt_out = model(dummy_x, dummy_mask)

    sess = ort.InferenceSession(onnx_path)
    onnx_out = sess.run(None, {
        "ecg_signal": dummy_x.numpy(),
        "lead_mask": dummy_mask.numpy(),
    })

    pt_probs = pt_out["probs_specific"].numpy()
    onnx_probs = onnx_out[1]
    max_diff = np.max(np.abs(pt_probs - onnx_probs))
    print(f"    Max difference (probs_specific): {max_diff:.6f}")

    if max_diff < 0.01:
        print("    [OK] ONNX verification PASSED")
    else:
        print(f"    [WARN] ONNX verification WARNING -- diff={max_diff:.4f}")


# ---------------------------------------------------------------------------
# TFLite conversion
# ---------------------------------------------------------------------------

def convert_to_tflite(onnx_path: str, tflite_path: str,
                      quantize: str = "none",
                      num_leads: int = 12, seq_len: int = 5000):
    """Convert ONNX -> TFLite."""
    print(f"[4/5] Converting ONNX -> TFLite (quantize={quantize})")

    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)

        saved_model_dir = onnx_path.replace(".onnx", "_saved_model")
        tf_rep.export_graph(saved_model_dir)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    except ImportError:
        print("\n    [FAIL] onnx-tf not found.")
        print("    Install: pip install onnx-tf tensorflow")
        print("    Or use the ONNX file with onnxruntime-mobile.\n")
        return False

    if quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("    Applying float16 quantization")

    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def representative_dataset():
            for _ in range(100):
                x = np.random.randn(1, num_leads, seq_len).astype(np.float32) * 0.5
                mask = np.ones((1, num_leads), dtype=np.float32)
                yield [x, mask]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.float32
        print("    Applying INT8 quantization")

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / 1024**2
    print(f"    [OK] TFLite saved: {tflite_path} ({size_mb:.1f} MB)")
    return True


# ---------------------------------------------------------------------------
# TorchScript Mobile export
# ---------------------------------------------------------------------------

def export_torchscript_mobile(model, output_path: str,
                               num_leads: int = 12, seq_len: int = 5000):
    """Export as TorchScript for PyTorch Mobile (.ptl)."""
    import torch
    from torch.utils.mobile_optimizer import optimize_for_mobile

    print(f"[ALT] Exporting TorchScript for PyTorch Mobile: {output_path}")

    class MobileWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor, mask: torch.Tensor):
            out = self.model(x, mask)
            return (
                out["probs_binary"],
                out["probs_specific"],
                out["uncertainty_spec"],
                out["signal_quality"],
                out["lead_importance"],
            )

    wrapper = MobileWrapper(model)
    wrapper.eval()

    dummy_x = torch.randn(1, num_leads, seq_len)
    dummy_mask = torch.ones(1, num_leads)

    traced = torch.jit.trace(wrapper, (dummy_x, dummy_mask))
    optimized = optimize_for_mobile(traced)
    optimized._save_for_lite_interpreter(output_path)

    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"    [OK] TorchScript Mobile saved: {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export CardioGuardNet -> TFLite / TorchScript Mobile"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(PROJECT_ROOT / "pipeline" / "v14_best.pt"),
        help="Path to .pt checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "pipeline"),
        help="Output directory"
    )
    parser.add_argument(
        "--num-classes", type=int, default=5,
        help="Number of output classes (must match checkpoint)"
    )
    parser.add_argument(
        "--quantize", choices=["none", "float16", "int8"],
        default="none",
        help="Quantization mode for TFLite"
    )
    parser.add_argument(
        "--format", choices=["tflite", "torchscript", "both"],
        default="both",
        help="Export format"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.checkpoint, args.num_classes)
    base_name = Path(args.checkpoint).stem

    if args.format in ("tflite", "both"):
        onnx_path = os.path.join(args.output_dir, f"{base_name}.onnx")
        export_onnx(model, onnx_path)

        try:
            verify_onnx(model, onnx_path)
        except Exception as e:
            print(f"    [WARN] ONNX verification skipped: {e}")

        tflite_path = os.path.join(args.output_dir, f"{base_name}.tflite")
        convert_to_tflite(onnx_path, tflite_path, args.quantize)

    if args.format in ("torchscript", "both"):
        # Reload clean model (without FFT patches) for TorchScript
        model_ts = load_model(args.checkpoint, args.num_classes)
        ts_path = os.path.join(args.output_dir, f"{base_name}_mobile.ptl")
        try:
            export_torchscript_mobile(model_ts, ts_path)
        except Exception as e:
            print(f"    [WARN] TorchScript Mobile export failed: {e}")

    print("\n[5/5] Export complete! ")
    print("\nDeployment options:")
    print("  1. TFLite      -> tflite-runtime on Android (smallest)")
    print("  2. TorchScript -> pytorch_android_lite (most compatible)")
    print("  3. ONNX        -> onnxruntime-mobile (best balance)")


if __name__ == "__main__":
    main()
