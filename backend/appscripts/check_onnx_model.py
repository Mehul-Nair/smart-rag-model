#!/usr/bin/env python3
"""
Check ONNX Model Specifications
"""

import onnx
import onnxruntime as ort
import numpy as np


def check_onnx_model():
    """Check ONNX model specifications"""

    model_path = "trained_deberta_ner_model/model.onnx"

    print(f"Checking ONNX model: {model_path}")

    try:
        # Load ONNX model
        model = onnx.load(model_path)
        print("✅ ONNX model loaded successfully")

        # Check model inputs
        print("\n📋 Model Inputs:")
        for input_info in model.graph.input:
            print(f"  Name: {input_info.name}")
            print(f"  Type: {input_info.type.tensor_type.elem_type}")
            print(
                f"  Shape: {[dim.dim_value for dim in input_info.type.tensor_type.shape.dim]}"
            )
            print()

        # Check model outputs
        print("📋 Model Outputs:")
        for output_info in model.graph.output:
            print(f"  Name: {output_info.name}")
            print(f"  Type: {output_info.type.tensor_type.elem_type}")
            print(
                f"  Shape: {[dim.dim_value for dim in output_info.type.tensor_type.shape.dim]}"
            )
            print()

        # Check with ONNX Runtime
        print("🔍 ONNX Runtime Session Info:")
        session = ort.InferenceSession(model_path)

        print("Inputs:")
        for input_meta in session.get_inputs():
            print(f"  Name: {input_meta.name}")
            print(f"  Type: {input_meta.type}")
            print(f"  Shape: {input_meta.shape}")
            print()

        print("Outputs:")
        for output_meta in session.get_outputs():
            print(f"  Name: {output_meta.name}")
            print(f"  Type: {output_meta.type}")
            print(f"  Shape: {output_meta.shape}")
            print()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    check_onnx_model()
