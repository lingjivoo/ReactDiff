# 🎭 ReactDiff: Fundamental Multiple Appropriate Facial Reaction Diffusion Model

<div align="center">
  
[![Project Page](https://img.shields.io/badge/🌐-Project%20Page-blue)](https://reactdiff.github.io)
[![Paper](https://img.shields.io/badge/📄-Paper%20arXiv-red)](https://arxiv.org/abs/2510.04712)
[![Code](https://img.shields.io/badge/💻-Code%20GitHub-black)](https://github.com/lingjivoo/ReactDiff)

https://github.com/user-attachments/assets/471100bc-5adb-4130-a866-8d964f538dc5

<div align="center">
  
https://github.com/user-attachments/assets/19fc7612-c7ee-46ad-9dd4-99fb2235b4ef  

https://github.com/user-attachments/assets/dc5f6f6e-4355-4302-a67d-920fefbdb32f  

https://github.com/user-attachments/assets/021356d9-d055-4716-bb1f-5a6274b3b899
</div>

<div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
  <div style="text-align: center;">
    <a href="https://reactdiff.github.io/assets/media/video-3.mp4" target="_blank">
      <img src="https://img.shields.io/badge/🎭-Watch%20Demo-blue?style=for-the-badge&logo=video" alt="Realistic Listener Reactions" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
    </a>
    <p style="margin-top: 8px; font-size: 14px; color: #666;">🎭 Realistic Listener Reactions</p>
  </div>
  <div style="text-align: center;">
    <a href="https://reactdiff.github.io/assets/media/video-2.mp4" target="_blank">
      <img src="https://img.shields.io/badge/🎬-Watch%20Demo-green?style=for-the-badge&logo=video" alt="30-Second Full Sequences" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
    </a>
    <p style="margin-top: 8px; font-size: 14px; color: #666;">🎬 30-Second Full Sequences</p>
  </div>
  <div style="text-align: center;">
    <a href="https://reactdiff.github.io/assets/media/video-4.mp4" target="_blank">
      <img src="https://img.shields.io/badge/⚡-Watch%20Demo-orange?style=for-the-badge&logo=video" alt="Real-Time Processing" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
    </a>
    <p style="margin-top: 8px; font-size: 14px; color: #666;">⚡ Real-Time Processing</p>
  </div>
</div>

</div>

<div align="center">
  
## 🎥 Demo Videos

**High-quality listener reaction generation with ReactDiff**

### 📺 Video 1: Realistic Listener Reactions
[![Watch Demo](https://img.shields.io/badge/🎭-Watch%20Realistic%20Listener%20Reactions-blue?style=for-the-badge&logo=video)](https://reactdiff.github.io/assets/media/video-3.mp4)

### 📺 Video 2: 30-Second Full Sequences  
[![Watch Demo](https://img.shields.io/badge/🎬-Watch%2030s%20Full%20Sequences-green?style=for-the-badge&logo=video)](https://reactdiff.github.io/assets/media/video-2.mp4)

### 📺 Video 3: Real-Time Processing
[![Watch Demo](https://img.shields.io/badge/⚡-Watch%20Real--Time%20Processing-orange?style=for-the-badge&logo=video)](https://reactdiff.github.io/assets/media/video-4.mp4)

---

**💡 Tip**: Click the buttons above to watch the demo videos in a new tab. The videos showcase ReactDiff's ability to generate realistic listener facial reactions from speaker audio and visual cues.

</div>

## 📢 News

- 🎉 **ReactDiff v1.0 Released!** A diffusion-based model for generating realistic listener facial reactions from speaker audio and visual cues! (Dec/2024)
- 🚀 **Multi-GPU Support** - Now supports distributed training and evaluation across multiple GPUs
- 🎬 **30-Second Video Generation** - Generate full-length realistic listener reaction videos
- 🔧 **Enhanced Configuration** - Separate configs for training and evaluation with detailed documentation

## 📋 Table of Contents

- [🛠️ Installation](#installation)
- [👨‍🏫 Getting Started](#getting-started)
  - [📊 Data Preparation](#1-data-preparation)
  - [🔧 External Tool Preparation](#2-external-tool-preparation)
- [⚙️ Configuration](#configuration)
- [🚀 Training](#training)
- [📊 Evaluation](#evaluation)
- [🎯 Key Features](#key-features)
- [📁 Project Structure](#project-structure)
- [🐛 Troubleshooting](#troubleshooting)
- [🖊️ Citation](#citation)
- [🤝 Acknowledgements](#acknowledgements)

## 🛠️ Installation

### 📋 Prerequisites

- 🐍 **Python 3.8+** - Required for modern Python features
- 🔥 **PyTorch 1.9+** - Deep learning framework
- ⚡ **CUDA 11.8+** - GPU acceleration support
- 💾 **16GB+ RAM** - Recommended for training
- 🎮 **NVIDIA GPU** - Required for CUDA acceleration

### 🚀 Quick Setup

#### 1️⃣ Create and activate conda environment
```bash
# Create a new conda environment with Python 3.9
conda create -n reactdiff python=3.9

# Activate the environment
conda activate reactdiff
```

#### 2️⃣ Install PyTorch with CUDA support
```bash
# Install PyTorch 2.0.1 with CUDA 11.8 support
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

#### 3️⃣ Install PyTorch3D for 3D operations
```bash
# Install PyTorch3D for 3D face model operations
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html
```

#### 4️⃣ Install all other dependencies
```bash
# Install all required packages from requirements.txt
pip install -r requirements.txt
```

### ✅ Verify Installation

```bash
# Test if all imports work correctly
python -c "
import torch
import torchvision
import numpy as np
import cv2
import transformers
print('✅ All dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
"
```

## 👨‍🏫 Getting Started

### 1. Data Preparation
<details>
<summary><b>Download and Setup Dataset</b></summary>

The REACT 2023/2024 Multimodal Challenge Dataset is compiled from the following public datasets for studying dyadic interactions:
- [NOXI](https://dl.acm.org/doi/10.1145/3136755.3136780)
- [RECOLA](https://ieeexplore.ieee.org/document/6553805)

Apply for data access at:
- [REACT 2023 Homepage](https://sites.google.com/cam.ac.uk/react2023/home)
- [REACT 2024 Homepage](https://sites.google.com/cam.ac.uk/react2024)

**Data organization (`data/`) follows this structure:**
```
data/partition/modality/site/chat_index/person_index/clip_index/actual_data_files
```

Example data structure:
```
data
├── test
├── val
├── train
   ├── Video_files
       ├── NoXI
           ├── 010_2016-03-25_Paris
               ├── Expert_video
               ├── Novice_video
                   ├── 1
                       ├── 1.png
                       ├── ....
                       ├── 751.png
                   ├── ....
           ├── ....
       ├── RECOLA
   ├── Audio_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.wav
                   ├── ....
           ├── group-2
           ├── group-3
   ├── Emotion
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.csv
                   ├── ....
           ├── group-2
           ├── group-3
   ├── 3D_FV_files
       ├── NoXI
       ├── RECOLA
           ├── group-1
               ├── P25 
               ├── P26
                   ├── 1.npy
                   ├── ....
           ├── group-2
           ├── group-3
```

Important details:
- Task: Predict one role's reaction ('Expert' or 'Novice', 'P25' or 'P26') to the other
- 3D_FV_files contain 3DMM coefficients (expression: 52 dim, angle: 3 dim, translation: 3 dim)
- Video specifications:
  - Frame rate: 25 fps
  - Resolution: 256x256
  - Clip length: 751 frames (~30s)
  - Audio sampling rate: 44100
- CSV files for training/validation are available at: 'data/train.csv', 'data/val.csv', 'data/test.csv'

</details>

<details>
<summary><b>Download Additional Resources</b></summary>

1. **Listener Reaction Neighbors**
   - Download the appropriate listener reaction neighbors dataset from [here](https://drive.google.com/drive/folders/1gi1yzP3dUti8fJy2HToiijPuyRyzokdh?usp=sharing)
   - Place the downloaded files in the dataset root folder
   - 
2. **Ground Truth 3DMMs**
   - Download the ground truth 3DMMs (test set) for speaker-listener evaluation from [here](https://drive.google.com/drive/folders/1jVi8ZWMiyynG6LsKJSaKj2fX-EavK11h?usp=drive_link)
   - Place the downloaded files in the `metric/gt` folder

</details>

---

### 2. External Tool Preparation
<details>
<summary><b>Required Models and Tools</b></summary>

We use 3DMM coefficients for 3D listener/speaker representation and 3D-to-2D frame rendering.

1. **3DMM Model Setup**
   - Download [FaceVerse version 2 model](https://github.com/LizhenWangT/FaceVerse) (faceverse_simple_v2.npy)
   - Place in `external/FaceVerse/data/`
   - Get pre-extracted data:
     - [3DMM coefficients](https://drive.google.com/drive/folders/1RrTytDkkq520qUUAjTuNdmS6tCHQnqFu) (Place in `dataset_root/3D_FV_files`)
     - [Reference files](https://drive.google.com/drive/folders/1uVOOJzY3p2XjDESwH4FCjGO8epO7miK4) (mean_face, std_face, reference_full)
     - Place in `external/FaceVerse/`

2. **PIRender Setup**
   - We use [PIRender](https://github.com/RenYurui/PIRender) for 3D-to-2D rendering
   - Download our retrained [checkpoint](https://drive.google.com/drive/folders/1Ys1u0jxVBxrmQZrcrQbm8tagOPNxrTUA) (cur_model_fold.pth)
   - Place in `external/PIRender/`

</details>

---

## ⚙️ Configuration

ReactDiff uses separate configuration files for different phases:

### Configuration Files

- **`configs/config_train.json`** - Training configuration (shorter sequences, augmentation enabled)
- **`configs/config_eval.json`** - Evaluation configuration (30-second sequences, PIRender enabled)
- **`configs/config.json`** - General configuration (legacy support)

### Key Parameters

- **Model**: 58D 3DMM input, U-Net architecture with cross-attention
- **Dataset**: Configurable sequence length (256 for training, 750 for evaluation)
- **Training**: AdamW optimizer, mixed precision FP16, gradient accumulation
- **Evaluation**: Window-based processing, PIRender integration, video generation

See [Configuration Documentation](configs/README.md) for detailed parameter descriptions.

---

## 🚀 Training

<details>
<summary><b>Training Options</b></summary>

### Single GPU Training
```bash
cd src/scripts
./run_train.sh single
```

### Multi-GPU Training
```bash
cd src/scripts
./run_train.sh multi
```

### Custom Training
```bash
python train.py \
  --config ../../configs/config_train.json \
  --out-path ./results/training \
  --name reactdiff_model \
  --batch-size 100 \
  --window-size 16 \
  --weight-kinematics-loss 0.01 \
  --weight-velocity-loss 1.0
```

**Training Features:**
- Mixed precision training (FP16)
- Gradient accumulation
- Weights & Biases logging
- EMA (Exponential Moving Average) for model weights
- Configurable loss weights for kinematics and velocity

</details>

---

## 📊 Evaluation

<details>
<summary><b>Generate Results</b></summary>

### Single GPU Evaluation
```bash
cd src/scripts
./run_eval.sh single /path/to/checkpoint.pth
```

### Multi-GPU Evaluation
```bash
cd src/scripts
./run_eval.sh multi /path/to/checkpoint.pth
```

### Custom Evaluation
```bash
python sample.py \
  --config ../../configs/config_eval.json \
  --checkpoint /path/to/checkpoint.pth \
  --out-path ./results/evaluation \
  --window-size 64 \
  --steps 50 \
  --momentum 0.9
```

**Evaluation Features:**
- Full 30-second video generation (750 frames)
- PIRender integration for realistic rendering
- Chunked processing for memory efficiency
- Configurable sampling parameters

</details>


---

## 🎯 Key Features

### 🧠 Diffusion Model Architecture
- **🎨 Karras et al. (2022) Framework** - State-of-the-art diffusion model implementation
- **🏗️ U-Net Backbone** - Robust architecture with cross-attention layers for multi-modal fusion
- **⏱️ Temporal Windowing** - Online inference capability for real-time applications
- **📐 3DMM Parameter Prediction** - 58-dimensional facial parameter generation (52 expression + 3 rotation + 3 translation)

### 🎵 Audio-Visual Processing
- **🎤 Wav2Vec2 Integration** - Pre-trained audio feature extraction for robust speech understanding
- **👤 3DMM Coefficients** - Comprehensive facial representation using 3D Morphable Model
- **🔗 Cross-Modal Attention** - Sophisticated attention mechanisms between audio and visual features
- **🎯 Multi-Scale Processing** - Hierarchical feature extraction for different temporal scales

### 🎬 Rendering Pipeline
- **🎨 PIRender Integration** - High-quality 3D-to-2D rendering for realistic video generation
- **👤 FaceVerse Processing** - Advanced 3DMM parameter processing and facial modeling
- **💾 Chunked Processing** - Memory-efficient processing for long sequences (30+ seconds)
- **📹 Configurable Output** - Flexible video output formats (MP4, AVI) with customizable quality

### 🚀 Training & Evaluation
- **⚙️ Separate Configurations** - Optimized settings for training vs. evaluation phases
- **⚡ Mixed Precision Training** - FP16 training for 2x speed improvement and memory efficiency
- **🖥️ Multi-GPU Support** - Distributed training and evaluation via HuggingFace Accelerate
- **📊 Comprehensive Logging** - Weights & Biases integration for experiment tracking
- **🔄 EMA (Exponential Moving Average)** - Stable model weights for better convergence

### 🎭 Advanced Capabilities
- **🎬 Full-Length Video Generation** - Generate complete 30-second reaction videos
- **🔄 Real-Time Processing** - Online inference with configurable window sizes
- **🎯 Multiple Reaction Styles** - Generate diverse and appropriate listener reactions
- **📈 Scalable Architecture** - Support for various sequence lengths and batch sizes
- **🛠️ Extensive Customization** - Highly configurable parameters for different use cases

---

## 🖊️ Citation

If this work helps in your research, please cite:

```bibtex
@article{cheng2025reactdiff,
  title={ReactDiff: Fundamental Multiple Appropriate Facial Reaction Diffusion Model},
  author={Cheng, Luo and Siyang, Song and Siyuan, Yan and Zhen, Yu and Zongyuan, Ge},
  journal={arXiv preprint arXiv:2510.04712},
  year={2025}
}
```

## 🤝 Acknowledgements

Thanks to the open source of the following projects:

- [Karras et al. (2022) Diffusion Models](https://github.com/crowsonkb/k-diffusion)
- [FaceVerse](https://github.com/LizhenWangT/FaceVerse)
- [PIRender](https://github.com/RenYurui/PIRender)
- [Wav2Vec2](https://github.com/huggingface/transformers)
- [Accelerate](https://github.com/huggingface/accelerate)

---

## 📁 Project Structure

```
ReactDiff/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── configs/                  # Configuration files
│   ├── config_train.json    # Training configuration
│   ├── config_eval.json     # Evaluation configuration
│   ├── config.json          # General configuration
│   └── README.md            # Configuration documentation
├── src/                     # Source code
│   ├── models/              # Model implementations
│   │   └── k_diffusion/     # Diffusion model components
│   ├── data/                # Data handling
│   │   └── dataset.py       # Dataset loading and preprocessing
│   ├── utils/               # Utility functions
│   │   ├── utils.py         # General utilities
│   │   ├── render.py        # 3DMM to video rendering
│   │   └── metric/          # Evaluation metrics
│   ├── external/            # External dependencies
│   │   ├── facebook/        # Wav2Vec2 model
│   │   ├── FaceVerse/       # FaceVerse components
│   │   └── PIRender/        # PIRender components
│   └── scripts/             # Training and evaluation scripts
│       ├── train.py         # Training script
│       ├── sample.py        # Main sampling script
│       ├── run_train.sh     # Training shell script
│       └── run_eval.sh      # Evaluation shell script
├── docs/                    # Documentation
└── results/                 # Output directory (created during training)
```

---

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and paths are correct
2. **CUDA OOM**: Reduce batch size or window size in configuration
3. **Short Videos**: Check `clip_length` in evaluation config (should be 750 for 30s)
4. **PIRender Errors**: Verify PIRender checkpoint and FaceVerse files are in place

### Performance Tips

1. Use appropriate configuration file for your task
2. Adjust batch size based on GPU memory
3. Use multi-GPU for large-scale training
4. Enable mixed precision for faster processing
5. Use chunked processing for long sequences

For more detailed troubleshooting, see the [Configuration Documentation](configs/README.md).


