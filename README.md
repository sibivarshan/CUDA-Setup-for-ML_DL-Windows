# CUDA installation for traing ml dl models in Laptop's GPU(Windows)

This repository provides a step-by-step guide to set up your environment for training machine learning (ML) and deep learning (DL) models using NVIDIA CUDA on a GPU. Follow these instructions to utilize the full power of your GPU for accelerated computation.

---

## Prerequisites

1. **Hardware**:
   - A system with an NVIDIA GPU (e.g., NVIDIA GeForce RTX 3050 Ti or similar).

2. **Software**:
   - Operating System: Windows 10/11
   - Python (version 3.8 or higher)
   - NVIDIA Drivers
   - CUDA Toolkit (version 12.x or compatible)
   - cuDNN Library

---

## Step-by-Step Setup

### 1. Install NVIDIA Drivers
Ensure you have the latest NVIDIA GPU drivers installed:
1. Download drivers from [NVIDIA’s official website](https://www.nvidia.com/Download/index.aspx).
2. Install the **Game Ready Driver** or **Studio Driver** as per your needs.
3. Verify installation:
   ```bash
   nvidia-smi
   ```
   This should display GPU details.

### 2. Install CUDA Toolkit

1. Download the CUDA Toolkit from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).
2. Install it and ensure the following paths are added to the system’s `Path` environment variable:
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\libnvvp`

3. Verify CUDA installation:
   ```bash
   nvcc --version
   ```
   This should show the installed CUDA version.

### 3. Install cuDNN

1. Download the cuDNN library (ZIP version) from [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn).
2. Extract and copy the files to the respective CUDA directories:
   - `bin\cudnn64_x.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
   - `include\cudnn.h` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include`
   - `lib\x64\cudnn.lib` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64`

3. Restart your system to apply changes.

### 4. Install Python and Required Libraries

1. Install Python from [python.org](https://www.python.org/downloads/).
2. Install TensorFlow or PyTorch with GPU support:
   - **TensorFlow**:
     ```bash
     pip install tensorflow
     ```
   - **PyTorch**:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```

3. Verify GPU availability in your chosen framework:
   - For TensorFlow:
     ```python
     import tensorflow as tf
     print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
     ```
   - For PyTorch:
     ```python
     import torch
     print("CUDA Available:", torch.cuda.is_available())
     print("Device Name:", torch.cuda.get_device_name(0))
     ```
     NOTE:Some versions of Tensorflow and PyTorch is not compatible with CUDA. Please check with it. 

---

## Sample Script to Test GPU

Run the following script to verify GPU functionality:

### TensorFlow:
```python
import tensorflow as tf

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using the GPU!")
```

### PyTorch:
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Perform a sample operation on the GPU
x = torch.rand(1000, 1000).to(device)
y = torch.rand(1000, 1000).to(device)
z = torch.matmul(x, y)
print("Operation completed on:", device)
```

---

## Troubleshooting

1. **`nvidia-smi` Works but TensorFlow/PyTorch Doesn't Detect GPU**:
   - Ensure CUDA and cuDNN versions are compatible with your framework.
   - Reinstall TensorFlow or PyTorch with GPU support.

2. **Environment Variables Not Set**:
   - Add the correct CUDA paths to your system's `Path`.

3. **Driver Installation Issues**:
   - Use Nvidia app for installing the driver or you can do it manually by following the step below. 
   - Use the [NVIDIA Driver Download Page](https://www.nvidia.com/Download/index.aspx) to reinstall.
   - NOTE: Prefer Studio Driver over Game ready driver for this purpose.

---

## Resources

- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN Library](https://developer.nvidia.com/cudnn)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)

---

## Author
Created by [Sibivarshan](https://github.com/sibivarshan). Contributions are welcome!
