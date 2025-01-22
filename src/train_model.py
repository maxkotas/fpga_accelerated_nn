# src/train_model.py
import os
import numpy as np
import tensorflow as tf
from qkeras import QDense, QConv2D, QActivation, quantizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, MaxPooling2D, Activation

# Enable eager execution
tf.config.run_functions_eagerly(True)

def build_quantized_model():
    """Defines the quantized neural network model using QKeras."""
    model = Sequential([
        # Quantized Conv2D layer
        QConv2D(16, kernel_size=(3, 3), 
                input_shape=(28, 28, 1), 
                kernel_quantizer=quantizers.quantized_bits(4, 2, 1),  # 4 bits, 2 fractional
                bias_quantizer=quantizers.quantized_bits(4, 2, 1),
                name="qconv1"),
        QActivation(activation=quantizers.quantized_relu(4), name="qrelu1"),  # Quantized ReLU activation
        MaxPooling2D(pool_size=(2, 2), name="pool1"),  # Standard pooling layer
        
        Flatten(name="flatten"),  # Flatten layer
        QDense(64, 
               kernel_quantizer=quantizers.quantized_bits(4, 2, 1), 
               bias_quantizer=quantizers.quantized_bits(4, 2, 1),
               name="qdense1"),
        QActivation(activation=quantizers.quantized_relu(4), name="qrelu2"),  # Quantized ReLU activation
        QDense(10, 
               kernel_quantizer=quantizers.quantized_bits(4, 2, 1), 
               bias_quantizer=quantizers.quantized_bits(4, 2, 1),
               name="qdense2"),
        Activation("softmax", name="softmax")  # Use standard softmax
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load preprocessed data
    data_path = os.path.join("..", "data", "mnist_data.npz")
    data = np.load(data_path)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    
    # Normalize the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Build and train the quantized model
    model = build_quantized_model()
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=32)
    
    # Save the trained quantized model
    model_save_path = os.path.join("..", "models", "mnist_model_qkeras.h5")
    model.save(model_save_path)
    print(f"Quantized model saved to {model_save_path}")