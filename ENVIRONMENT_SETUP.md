# 🛠️ Environment Setup Guide

This comprehensive guide will help you set up the ReactDiff environment from scratch, including all dependencies, external tools, and configurations.

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Environment Setup](#environment-setup)
- [External Dependencies](#external-dependencies)
- [Configuration Setup](#configuration-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Ubuntu 18.04+ / Windows 10+ / macOS 10.15+
- **Python**: 3.8+ (3.9 recommended)
- **RAM**: 16GB (32GB recommended for training)
- **Storage**: 50GB free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080+ recommended)

### Recommended Setup
- **OS**: Ubuntu 20.04 LTS
- **Python**: 3.9.7
- **RAM**: 32GB+
- **Storage**: 100GB+ SSD
- **GPU**: NVIDIA RTX 4090 / A100 (for training)
- **CUDA**: 11.8+

## 🚀 Environment Setup

### 1️⃣ System Preparation

#### Ubuntu/Debian
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential cmake git wget curl

# Install Python development headers
sudo apt install -y python3-dev python3-pip

# Install FFmpeg for video processing
sudo apt install -y ffmpeg

# Install OpenCV dependencies
sudo apt install -y libopencv-dev python3-opencv
```

#### Windows
```powershell
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install Git for Windows
# Download from: https://git-scm.com/download/win

# Install FFmpeg
# Download from: https://ffmpeg.org/download.html
```

#### macOS
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install essential tools
brew install cmake git wget curl

# Install Python
brew install python@3.9

# Install FFmpeg
brew install ffmpeg
```

### 2️⃣ CUDA Installation

#### Install CUDA Toolkit 11.8
```bash
# Download CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# Install CUDA
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
nvidia-smi
```

### 3️⃣ Conda Environment Setup

```bash
# Install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create ReactDiff environment
conda create -n reactdiff python=3.9 -y
conda activate reactdiff

# Verify Python version
python --version  # Should show Python 3.9.x
```

### 4️⃣ PyTorch Installation

```bash
# Activate environment
conda activate reactdiff

# Install PyTorch with CUDA 11.8 support
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

### 5️⃣ PyTorch3D Installation

```bash
# Install PyTorch3D for 3D operations
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html

# Verify PyTorch3D installation
python -c "
import pytorch3d
print(f'PyTorch3D version: {pytorch3d.__version__}')
"
```

### 6️⃣ Install ReactDiff Dependencies

```bash
# Clone ReactDiff repository
git clone https://github.com/your-username/ReactDiff.git
cd ReactDiff

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "
import torch
import torchvision
import numpy as np
import cv2
import transformers
import accelerate
import wandb
import tqdm
import skvideo
import skimage
print('✅ All dependencies installed successfully!')
"
```

## 🔌 External Dependencies

### 1️⃣ Wav2Vec2 Setup

```bash
# The Wav2Vec2 model will be automatically downloaded on first use
# Manual download (optional):
cd src/external/facebook/
wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/pytorch_model.bin
wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json
```

### 2️⃣ FaceVerse Setup

```bash
# Download FaceVerse model
cd src/external/FaceVerse/
wget https://github.com/LizhenWangT/FaceVerse/releases/download/v2.0/faceverse_simple_v2.npy

# Download reference files
wget https://drive.google.com/uc?id=YOUR_MEAN_FACE_ID -O mean_face.npy
wget https://drive.google.com/uc?id=YOUR_STD_FACE_ID -O std_face.npy
wget https://drive.google.com/uc?id=YOUR_REFERENCE_FULL_ID -O reference_full.npy
```

### 3️⃣ PIRender Setup

```bash
# Download PIRender checkpoint
cd src/external/PIRender/
wget https://drive.google.com/uc?id=YOUR_PIRENDER_CHECKPOINT_ID -O cur_model_fold.pth
```

### 4️⃣ Dataset Setup

```bash
# Create dataset directory structure
mkdir -p data/{train,val,test}/{Video_files,Audio_files,Emotion,3D_FV_files}

# Download REACT 2023/2024 dataset
# Apply for access at: https://sites.google.com/cam.ac.uk/react2024

# Organize data according to the structure in README.md
```

## ⚙️ Configuration Setup

### 1️⃣ Basic Configuration

```bash
# Copy configuration templates
cp configs/config_template.json configs/my_config.json

# Edit configuration files
nano configs/config_train.json    # Training configuration
nano configs/config_eval.json     # Evaluation configuration
```

### 2️⃣ Dataset Configuration

Update the dataset path in your configuration:

```json
{
    "dataset": {
        "location": "/path/to/your/dataset",
        "img_size": 256,
        "crop_size": 224,
        "clip_length": 750
    }
}
```

### 3️⃣ GPU Configuration

For multi-GPU setups:

```bash
# Check available GPUs
nvidia-smi

# Set CUDA_VISIBLE_DEVICES for specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Or use specific GPU IDs in scripts
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/config_train.json
```

## ✅ Verification

### 1️⃣ Complete System Check

```bash
# Run comprehensive verification
python -c "
import sys
import torch
import torchvision
import numpy as np
import cv2
import transformers
import accelerate
import wandb
import tqdm
import skvideo
import skimage
import pytorch3d

print('=== System Information ===')
print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

print('\n=== Package Versions ===')
packages = [
    'torch', 'torchvision', 'numpy', 'cv2', 'transformers', 
    'accelerate', 'wandb', 'tqdm', 'skvideo', 'skimage', 'pytorch3d'
]
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'Unknown')
        print(f'{pkg}: {version}')
    except ImportError:
        print(f'{pkg}: NOT INSTALLED')

print('\n✅ System verification completed!')
"
```

### 2️⃣ ReactDiff Import Test

```bash
# Test ReactDiff imports
cd src/scripts
python -c "
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import k_diffusion as K
from utils.render import Render
from data.dataset import get_dataloader
from external.wav2vec2focctc import Wav2Vec2ForCTC

print('✅ ReactDiff imports successful!')
print('✅ All modules loaded correctly!')
"
```

### 3️⃣ Configuration Test

```bash
# Test configuration loading
python -c "
import json
import sys
import os

# Test training config
with open('configs/config_train.json', 'r') as f:
    train_config = json.load(f)
print('✅ Training config loaded successfully!')

# Test evaluation config
with open('configs/config_eval.json', 'r') as f:
    eval_config = json.load(f)
print('✅ Evaluation config loaded successfully!')

print('✅ Configuration files are valid!')
"
```

## 🐛 Troubleshooting

### Common Issues

#### 1️⃣ CUDA Issues
```bash
# Problem: CUDA not found
# Solution: Check CUDA installation
nvcc --version
nvidia-smi

# Problem: PyTorch can't see CUDA
# Solution: Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

#### 2️⃣ Import Errors
```bash
# Problem: ModuleNotFoundError
# Solution: Check Python path and install missing packages
pip install -r requirements.txt

# Problem: External module not found
# Solution: Check external dependencies are downloaded
ls src/external/FaceVerse/
ls src/external/PIRender/
```

#### 3️⃣ Memory Issues
```bash
# Problem: CUDA out of memory
# Solution: Reduce batch size in config
# Edit configs/config_train.json:
# "batch_size": 32  # Reduce from 100

# Problem: System out of memory
# Solution: Use gradient accumulation
# "grad_accum_steps": 4  # Increase from 1
```

#### 4️⃣ Video Processing Issues
```bash
# Problem: FFmpeg not found
# Solution: Install FFmpeg
sudo apt install ffmpeg  # Ubuntu
brew install ffmpeg      # macOS

# Problem: OpenCV issues
# Solution: Reinstall OpenCV
pip uninstall opencv-python
pip install opencv-python-headless
```

### Performance Optimization

#### 1️⃣ GPU Memory Optimization
```bash
# Use mixed precision training
# In config: "mixed_precision": "fp16"

# Use gradient accumulation
# In config: "grad_accum_steps": 4

# Use smaller batch sizes
# In config: "batch_size": 32
```

#### 2️⃣ CPU Optimization
```bash
# Set number of workers
# In config: "num_workers": 4

# Use pin memory
# In config: "pin_memory": true
```

#### 3️⃣ Multi-GPU Setup
```bash
# Use Accelerate for multi-GPU
accelerate config

# Launch with multiple GPUs
accelerate launch --multi_gpu train.py --config configs/config_train.json
```

## 📚 Additional Resources

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/)
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/)
- [Weights & Biases](https://docs.wandb.ai/)
- [OpenCV Documentation](https://docs.opencv.org/)

### Community
- [PyTorch Forums](https://discuss.pytorch.org/)
- [HuggingFace Community](https://huggingface.co/community)
- [GitHub Issues](https://github.com/your-username/ReactDiff/issues)

### Support
- Create an issue on GitHub for bugs
- Check existing issues for solutions
- Join our community discussions

---

## 🎉 Next Steps

After completing the environment setup:

1. **📖 Read** [GETTING_STARTED.md](GETTING_STARTED.md) for step-by-step usage
2. **🔧 Configure** your dataset paths in the configuration files
3. **🚀 Start** with the training or evaluation scripts
4. **📊 Monitor** your experiments with Weights & Biases

Happy coding! 🎭✨