# 🚀 FPGA Accelerated Neural Network with HLS4ML 🖥️🔧

Welcome to the **FPGA Accelerated Neural Network** project! This repository demonstrates how to design, train, and convert a QKeras-based neural network for FPGA deployment using **HLS4ML**, enabling lightning-fast and power-efficient inference on hardware.  

---

## 📂 Project Structure

    `bash
    fpga_accelerated_nn/
    ├── data/                    # Dataset and testbench data
    │   ├── mnist_data.npz       # Preprocessed MNIST dataset
    │   ├── input_features.dat   # Testbench input features
    │   ├── output_predictions.dat # Testbench expected outputs
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
    ├── tests/                   # Unit tests for Python scripts
    │   ├── test_preprocessing.py # Tests for prep_data.py
    │   ├── test_training.py      # Tests for train_model.py
    │   ├── test_conversion.py    # Tests for convert_model.py
    ├── README.md                # This README file!
    ├── LICENSE                  # License for the project
    ├── environment.yml          # Conda environment configuration
    └── .gitignore               # Git ignore rules
    `
---

## 🛠️ Installation

1. **Clone the repository**:
   `bash
   git clone https://github.com/yourusername/fpga_accelerated_nn.git
   cd fpga_accelerated_nn
   `

2. **Set up the Conda environment**:
   `bash
   conda env create -f environment.yml
   conda activate fpga-accelerator
   `

3. **Verify the installation**:
   `bash
   python --version  # Should match the Python version in environment.yml
   `

---

## 📖 Usage Guide

### 1. **Prepare the Dataset**
   Preprocess the MNIST dataset and save it for training and testing:
   `bash
   python src/prep_data.py
   `

### 2. **Train the Neural Network**
   Train a quantized QKeras model:
   `bash
   python src/train_model.py
   `
   The trained model will be saved in the `models/` directory.

### 3. **Convert the Model to HLS**
   Convert the trained QKeras model to an HLS4ML project:
   `bash
   python src/convert_model.py
   `
   The HLS project files will be saved in `hls4ml_model_qkeras/`.

### 4. **Perform HLS Synthesis**
   If you have Vivado HLS installed, you can synthesize the design:
   `bash
   python src/hls_synthesis.py
   `

---

## 🔬 Testing
Unit tests are included to validate each component of the pipeline:
- Run all tests:
  `bash
  pytest tests/
  `

---

## 🚀 Key Features

- **Quantized Neural Networks**: Leverage QKeras to design ultra-efficient models.
- **FPGA-Ready Designs**: Convert models to C++/HLS using HLS4ML.
- **Dataset Preprocessing**: Streamlined MNIST dataset preparation.
- **Modular Codebase**: Organized for ease of use and collaboration.
- **Vivado HLS Integration**: Synthesize directly if Vivado HLS is installed.

---

## 📊 Results
Once synthesized, the HLS project will provide:
- **Resource Utilization**: LUTs, DSPs, BRAMs.
- **Latency Estimates**: Model latency in clock cycles.
- **Throughput**: Predictions per second.

To simulate the design, check out the testbench in `hls4ml_model_qkeras/myproject_test.cpp`.

---

## 🎨 Contributing
Contributions are welcome! If you have ideas, bug fixes, or enhancements:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/my-new-feature`.
3. Commit your changes: `git commit -m "Add some feature"`.
4. Push to the branch: `git push origin feature/my-new-feature`.
5. Open a pull request.

---

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🧩 Related Projects
- [HLS4ML](https://fastmachinelearning.org/hls4ml/)
- [QKeras](https://github.com/google/qkeras)
- [Vivado HLS](https://www.xilinx.com/products/design-tools/vivado.html)

---

## ⭐ Acknowledgments
This project was inspired by the power of hardware acceleration and the growing need for efficient AI on the edge. Thanks to the [HLS4ML](https://fastmachinelearning.org/hls4ml/) team for their incredible tools!

---

Made with ❤️ and a touch of FPGA magic. ✨