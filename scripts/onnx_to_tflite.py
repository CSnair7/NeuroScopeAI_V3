#!/usr/bin/env python
"""
Convert existing ONNX model to TFLite.
Run this from an environment with onnx-tf and tensorflow installed.

Usage:
    python scripts/onnx_to_tflite.py
    python scripts/onnx_to_tflite.py --quantize float16
    python scripts/onnx_to_tflite.py --quantize int8
"""
import argparse
import os
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def convert(onnx_path, tflite_path, quantize="none"):
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    print(f"Loading ONNX: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    print(f"  Inputs:  {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")

    print("Converting ONNX -> TF SavedModel...")
    tf_rep = prepare(onnx_model)
    saved_model_dir = onnx_path.replace(".onnx", "_saved_model")
    tf_rep.export_graph(saved_model_dir)
    print(f"  SavedModel: {saved_model_dir}")

    print("Converting SavedModel -> TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("  Quantization: float16")
    elif quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        def representative_dataset():
            for _ in range(100):
                x = np.random.randn(1, 12, 5000).astype(np.float32) * 0.5
                mask = np.ones((1, 12), dtype=np.float32)
                yield [x, mask]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.float32
        print("  Quantization: int8")

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / 1024**2
    print(f"  TFLite saved: {tflite_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default=str(PROJECT_ROOT / "pipeline" / "v14_best.onnx"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "pipeline" / "v14_best.tflite"))
    parser.add_argument("--quantize", choices=["none", "float16", "int8"], default="none")
    args = parser.parse_args()
    convert(args.onnx, args.output, args.quantize)


if __name__ == "__main__":
    main()
