# 🚀 FPGA Accelerated Neural Network with HLS4ML 🖥️🔧

Welcome to the **FPGA Accelerated Neural Network** project! This repository demonstrates how to design, train, and convert a QKeras-based neural network for FPGA deployment using **HLS4ML**, enabling lightning-fast and power-efficient inference on hardware.

---

## 📂 Project Structure

    fpga_accelerated_nn/
    ├── data/                    # Dataset and testbench data
    │   ├── mnist_data.npz       # Preprocessed MNIST dataset
    ├── src/                     # Python scripts for core functionality
    │   ├── prep_data.py         # Prepare the MNIST dataset
    │   ├── train_model.py       # Train a quantized QKeras model
    │   ├── convert_model.py     # Convert QKeras model to HLS
    │   ├── hls_synthesis.py     # Perform HLS synthesis
    ├── models/                  # Saved models
    │   ├── mnist_model_qkeras.h5 # Trained QKeras model
    ├── hls4ml_model_qkeras/     # HLS4ML generated project directory
    │   ├── firmware/            # Generated HLS C++ code
    │   ├── tb_data/             # Testbench data files
    │   ├── hls4ml_config.yml    # HLS4ML configuration
    │   ├── myproject_test.cpp   # Testbench C++ implementation
    │   ├── vivado_synth.tcl     # Vivado synthesis script
    ├── README.md                # This README file!
    ├── LICENSE                  # License for the project
    ├── environment.yml          # Conda environment configuration
    └── .gitignore               # Git ignore rules

---

## ⚡ Key Features

- **QKeras**: We use QKeras to quantize neural network layers to 4-bit precision, enabling faster inference on FPGA.
- **HLS4ML**: Converts Keras models into hardware-friendly formats for FPGA acceleration.
- **MNIST Dataset**: Utilizes the famous MNIST dataset for training and testing the neural network.
- **Vivado HLS**: After generating the HLS model, we perform synthesis using Vivado HLS to generate the bitstream for FPGA deployment.

---

## 🛠️ Setup and Installation

1. Clone this repository:

    `bash
    git clone https://github.com/your-repo/fpga_accelerated_nn.git
    cd fpga_accelerated_nn
    

2. Install dependencies (Python 3.7+ required):

    `bash
    pip install -r requirements.txt
    

3. Make sure **Vivado HLS** is installed and accessible in your system’s PATH.

---

## 🧑‍💻 Usage

### 1. **Data Preprocessing**

First, preprocess the MNIST dataset. This will reshape the images and quantize them to integers.

Run the following to generate the preprocessed data:

`bash
python src/prep_data.py


This will create the `mnist_data.npz` file inside the `data/` directory.

### 2. **Train the Quantized Model**

Now, train the quantized model using the preprocessed data. This will generate a quantized neural network using QKeras.

Run the training script:

`bash
python src/train_model.py


The trained model will be saved as `mnist_model_qkeras.h5` inside the `models/` directory.

### 3. **Convert to HLS4ML Model**

Next, convert the trained Keras model into a hardware-friendly HLS4ML model. Run:

`bash
python src/convert_model.py


This will generate the HLS4ML model files in the `hls4ml_model_qkeras/` directory.

### 4. **Run HLS Synthesis**

Finally, run the Vivado HLS synthesis to generate the hardware bitstream:

`bash
python src/hls_synthesis.py


Make sure you have the Vivado HLS project files in place for synthesis.

---

## 📝 Files Explained

- **src/prep_data.py**: This script loads and preprocesses the MNIST dataset, quantizes the images, and saves them to disk.
- **src/train_model.py**: Defines, compiles, and trains the quantized model using QKeras.
- **src/convert_model.py**: Converts the trained QKeras model to HLS4ML format.
- **src/hls_synthesis.py**: Runs the Vivado HLS synthesis to generate FPGA bitstream files for hardware deployment.

---

## 🤖 Model Architecture

The neural network model consists of:

- **Quantized Conv2D**: Convolutional layer with quantized weights and biases.
- **Quantized ReLU**: Activation function with 4-bit precision.
- **MaxPooling2D**: Standard max-pooling layer.
- **Quantized Dense**: Fully connected layer with quantized weights and biases.
- **Softmax**: Output layer for classification.

---

## 📈 Training Results

    Model Summary:
    _________________________
    Layer (type)               Output Shape              Param #   
    ================================================================
    qconv1 (QConv2D)           (None, 26, 26, 16)        160
    qrelu1 (QActivation)       (None, 26, 26, 16)        0
    pool1 (MaxPooling2D)       (None, 13, 13, 16)        0
    flatten (Flatten)          (None, 2704)              0
    qdense1 (QDense)           (None, 64)                173120
    qrelu2 (QActivation)       (None, 64)                0
    qdense2 (QDense)           (None, 10)                650
    softmax (Activation)       (None, 10)                0
    ================================================================
    Total params: 173,930
    Trainable params: 173,930
    Non-trainable params: 0
    _________________________


---

## 💡 Contributing

Contributions are welcome! Please fork this repository, make changes, and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.