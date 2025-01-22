# src/convert_model.py
import os
import hls4ml
from tensorflow.keras.models import load_model
from qkeras import QDense, QConv2D, QActivation, quantizers
from tensorflow.keras.utils import custom_object_scope

def convert_to_hls_qkeras(model_path):
    """Converts a QKeras model to an HLS4ML model."""
    custom_objects = {
        "QConv2D": QConv2D,
        "QDense": QDense,
        "QActivation": QActivation,
        "quantized_bits": quantizers.quantized_bits,
        "quantized_relu": quantizers.quantized_relu,
    }

    # Ensure correct deserialization of the QKeras model
    with custom_object_scope(custom_objects):
        print(f"Loading model from: {model_path}")
        model = load_model(model_path, compile=False)

    # Generate HLS4ML configuration and convert
    print("Generating HLS4ML configuration...")
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    print("Generated HLS4ML configuration:", hls_config)

    print("Converting Keras model to HLS4ML...")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, 
        hls_config=hls_config, 
        output_dir='../hls4ml_model_qkeras'
    )
    hls_model.write()
    print("HLS4ML model files generated in ../hls4ml_model_qkeras directory.")

if __name__ == "__main__":
    model_path = os.path.join("..", "models", "mnist_model_qkeras.h5")
    convert_to_hls_qkeras(model_path)