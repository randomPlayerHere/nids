#!/usr/bin/env python3
"""
Convert Keras H5 model to TensorFlow Lite with optional quantization.

This script converts the NIDS DCNN model to TFLite format for:
- Lower memory footprint (~4x smaller)
- Faster CPU inference
- Better suited for resource-constrained deployments (Koyeb free tier, etc.)

Usage:
    python scripts/convert_to_tflite.py [--quantize fp16|int8|dynamic]

Options:
    --quantize fp16    Use FP16 quantization (2x smaller, minimal accuracy loss)
    --quantize int8    Use INT8 quantization (4x smaller, may lose some accuracy)
    --quantize dynamic Use dynamic range quantization (good balance)
    (default)          No quantization (full FP32)

Output:
    models/nids_dcnn_model.tflite
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─── Paths ───────────────────────────────────────────────────────────────────
H5_MODEL_PATH = "models/nids_dcnn_model.h5"
TFLITE_OUTPUT_PATH = "models/nids_dcnn_model.tflite"
SCALER_PATH = "models/cicids_scaler.pkl"


def get_representative_dataset(num_samples=100):
    """
    Generate representative data for INT8 calibration.
    Uses random data with similar distribution to CICIDS features.
    """
    import joblib
    
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        n_features = scaler.n_features_in_
    else:
        n_features = 78  # Default CICIDS feature count
    
    def representative_data_gen():
        for _ in range(num_samples):
            # Generate random data in typical CICIDS range
            data = np.random.randn(1, n_features, 1).astype(np.float32)
            yield [data]
    
    return representative_data_gen


def convert_to_tflite(quantize: str = None):
    """
    Convert Keras H5 model to TFLite.
    
    Args:
        quantize: Quantization mode - 'fp16', 'int8', 'dynamic', or None
    """
    print(f"📦 Loading Keras model from {H5_MODEL_PATH}...")
    
    if not os.path.exists(H5_MODEL_PATH):
        print(f"❌ Model not found: {H5_MODEL_PATH}")
        sys.exit(1)
    
    model = load_model(H5_MODEL_PATH)
    print(f"✅ Model loaded. Input shape: {model.input_shape}")
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply quantization if requested
    if quantize == "fp16":
        print("🔧 Applying FP16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
    elif quantize == "int8":
        print("🔧 Applying INT8 quantization (requires calibration data)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = get_representative_dataset()
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
    elif quantize == "dynamic":
        print("🔧 Applying dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    else:
        print("🔧 No quantization (full FP32)...")
    
    # Convert
    print("🔄 Converting to TFLite...")
    tflite_model = converter.convert()
    
    # Save
    os.makedirs(os.path.dirname(TFLITE_OUTPUT_PATH), exist_ok=True)
    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)
    
    # Report sizes
    h5_size = os.path.getsize(H5_MODEL_PATH) / (1024 * 1024)
    tflite_size = os.path.getsize(TFLITE_OUTPUT_PATH) / (1024 * 1024)
    reduction = (1 - tflite_size / h5_size) * 100
    
    print(f"\n✅ Conversion complete!")
    print(f"   H5 model size:     {h5_size:.2f} MB")
    print(f"   TFLite model size: {tflite_size:.2f} MB")
    print(f"   Size reduction:    {reduction:.1f}%")
    print(f"   Output: {TFLITE_OUTPUT_PATH}")
    
    # Verify the model works
    print("\n🧪 Verifying TFLite model...")
    verify_tflite_model()


def verify_tflite_model():
    """Quick verification that the TFLite model can run inference."""
    import joblib
    
    interpreter = tf.lite.Interpreter(model_path=TFLITE_OUTPUT_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape:  {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Input dtype:  {input_details[0]['dtype']}")
    print(f"   Output dtype: {output_details[0]['dtype']}")
    
    # Run a test inference
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        n_features = scaler.n_features_in_
    else:
        n_features = 78
    
    test_input = np.random.randn(1, n_features, 1).astype(np.float32)
    
    # Resize input tensor if needed
    interpreter.resize_tensor_input(input_details[0]['index'], test_input.shape)
    interpreter.allocate_tensors()
    
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"   Test output:  {output}")
    print(f"   ✅ Model verification passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Keras H5 model to TFLite with optional quantization"
    )
    parser.add_argument(
        "--quantize",
        choices=["fp16", "int8", "dynamic"],
        default=None,
        help="Quantization mode: fp16, int8, dynamic, or none (default)"
    )
    
    args = parser.parse_args()
    convert_to_tflite(args.quantize)


if __name__ == "__main__":
    main()
