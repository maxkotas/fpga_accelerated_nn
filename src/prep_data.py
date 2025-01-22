# src/prep_data.py
import os
import numpy as np
from tensorflow.keras.datasets import mnist

def load_and_preprocess_mnist():
    """Loads and preprocesses the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('uint8')  # Quantize to integers
    x_test = x_test.reshape(-1, 28, 28, 1).astype('uint8')    # Quantize to integers
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    # Save to the data/ directory
    output_path = os.path.join("..", "data", "mnist_data.npz")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    np.savez_compressed(output_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(f"Quantized MNIST data saved to {output_path}")