# 📁 ReactDiff Project Structure

This document provides a comprehensive overview of the ReactDiff project structure, explaining the purpose and organization of each directory and file.

## 🏗️ Root Directory Structure

```
ReactDiff/
├── 📄 README.md                    # Main project documentation
├── 📄 PROJECT_STRUCTURE.md         # This file - project structure documentation
├── 📄 ENVIRONMENT_SETUP.md         # Detailed environment setup guide
├── 📄 GETTING_STARTED.md           # Step-by-step getting started guide
├── 📄 requirements.txt             # Python dependencies
├── 📄 install.sh                   # Automated installation script
├── 📁 configs/                     # Configuration files
├── 📁 src/                         # Source code
├── 📁 docs/                        # Additional documentation
├── 📁 examples/                    # Example scripts and demos
└── 📁 results/                     # Output directory (created during execution)
```

## 📁 Configuration Directory (`configs/`)

```
configs/
├── 📄 README.md                    # Configuration documentation
├── 📄 config.json                  # General configuration (legacy)
├── 📄 config_train.json            # Training-specific configuration
├── 📄 config_eval.json             # Evaluation-specific configuration
└── 📄 config_template.json         # Template for custom configurations
```

### 🎯 Purpose
- **`config_train.json`** - Optimized for training with shorter sequences, augmentation enabled
- **`config_eval.json`** - Optimized for evaluation with 30-second sequences, PIRender enabled
- **`config.json`** - Legacy configuration for backward compatibility

## 📁 Source Code Directory (`src/`)

```
src/
├── 📁 models/                      # Model implementations
│   └── 📁 k_diffusion/            # Diffusion model components
├── 📁 data/                        # Data handling and preprocessing
├── 📁 utils/                       # Utility functions and helpers
├── 📁 external/                    # External dependencies and models
└── 📁 scripts/                     # Training and evaluation scripts
```

### 🧠 Models Directory (`src/models/`)

```
src/models/
└── 📁 k_diffusion/                 # Karras et al. (2022) diffusion framework
    ├── 📄 __init__.py             # Package initialization
    ├── 📄 Diffusion.py             # Main diffusion model class
    ├── 📄 layers.py                # Neural network layers and Denoiser
    ├── 📄 sampling.py              # Sampling algorithms and strategies
    ├── 📄 config.py                # Model configuration utilities
    ├── 📄 utils.py                 # Model utility functions
    ├── 📄 augmentation.py          # Data augmentation techniques
    ├── 📄 losses.py                # Loss functions
    └── 📄 gns.py                   # Gradient noise scale utilities
```

**Key Components:**
- **`Diffusion.py`** - Main ReactDiff model implementation
- **`layers.py`** - U-Net architecture with cross-attention layers
- **`sampling.py`** - DPM-Solver++ and other sampling methods
- **`config.py`** - Model configuration and parameter management

### 📊 Data Directory (`src/data/`)

```
src/data/
├── 📄 dataset.py                   # Main dataset class and data loading
├── 📄 __init__.py                  # Package initialization
└── 📁 __pycache__/                # Python cache files
```

**Key Features:**
- **`ReactionDataset`** - Main dataset class for loading REACT 2023/2024 data
- **Video validation** - Robust video file validation and error handling
- **3DMM processing** - 3D Morphable Model parameter loading and preprocessing
- **Audio processing** - Wav2Vec2 audio feature extraction
- **Multi-modal loading** - Coordinated loading of video, audio, and 3DMM data

### 🛠️ Utils Directory (`src/utils/`)

```
src/utils/
├── 📄 utils.py                     # General utility functions
├── 📄 render.py                    # 3DMM to video rendering pipeline
└── 📁 metric/                      # Evaluation metrics
    ├── 📄 __init__.py             # Package initialization
    └── 📄 metric.py               # Evaluation metric implementations
```

**Key Components:**
- **`render.py`** - Complete rendering pipeline with PIRender integration
- **`utils.py`** - Image processing, tensor operations, and helper functions
- **`metric/`** - Evaluation metrics for model performance assessment

### 🔌 External Directory (`src/external/`)

```
src/external/
├── 📄 wav2vec2focctc.py           # Wav2Vec2 model wrapper
├── 📄 reference_full.npy          # Reference 3DMM parameters
├── 📁 facebook/                    # Facebook Wav2Vec2 models
│   └── 📁 wav2vec2-base-960h/     # Pre-trained Wav2Vec2 model
├── 📁 FaceVerse/                   # FaceVerse 3D face model
│   ├── 📄 __init__.py             # Package initialization
│   ├── 📄 FaceVerseModel.py       # FaceVerse model implementation
│   ├── 📄 ModelRenderer.py        # 3D face rendering
│   ├── 📄 mean_face.npy           # Mean face parameters
│   ├── 📄 std_face.npy            # Standard deviation face parameters
│   └── 📄 LICENSE                  # FaceVerse license
└── 📁 PIRender/                    # PIRender for 3D-to-2D rendering
    ├── 📄 __init__.py             # Package initialization
    ├── 📄 base_function.py        # Base rendering functions
    ├── 📄 face_model.py           # Face model implementation
    ├── 📄 flow_util.py            # Optical flow utilities
    ├── 📄 cur_model_fold.pth      # Pre-trained PIRender checkpoint
    └── 📄 LICENSE.md              # PIRender license
```

**External Dependencies:**
- **Wav2Vec2** - Audio feature extraction
- **FaceVerse** - 3D face modeling and parameter processing
- **PIRender** - High-quality 3D-to-2D rendering

### 🚀 Scripts Directory (`src/scripts/`)

```
src/scripts/
├── 📄 train.py                     # Main training script
├── 📄 sample.py                    # Main evaluation/sampling script
├── 📄 run_train.sh                 # Training shell script (single/multi-GPU)
├── 📄 run_eval.sh                  # Evaluation shell script (single/multi-GPU)
└── 📁 __pycache__/                # Python cache files
```

**Scripts Overview:**
- **`train.py`** - Complete training pipeline with multi-GPU support
- **`sample.py`** - Evaluation and video generation pipeline
- **`run_train.sh`** - Convenient training script with configuration options
- **`run_eval.sh`** - Convenient evaluation script with rendering options

## 📁 Documentation Directory (`docs/`)

```
docs/
└── 📄 GETTING_STARTED.md           # Detailed getting started guide
```

## 📁 Examples Directory (`examples/`)

```
examples/
├── 📄 example_training.py          # Training example script
├── 📄 example_evaluation.py        # Evaluation example script
└── 📄 custom_inference.py          # Custom inference example
```

## 📁 Results Directory (`results/`)

```
results/                            # Created during execution
├── 📁 training/                    # Training outputs
│   ├── 📁 checkpoints/            # Model checkpoints
│   ├── 📁 logs/                   # Training logs
│   └── 📁 wandb/                  # Weights & Biases logs
├── 📁 evaluation/                  # Evaluation outputs
│   ├── 📁 video/                  # Generated videos
│   ├── 📁 coeffs/                 # 3DMM coefficients
│   └── 📁 metrics/                # Evaluation metrics
└── 📁 full_30s_test/              # Full sequence test outputs
```

## 🔄 Data Flow

### Training Flow
```
Raw Data → Dataset → Model → Loss → Optimizer → Checkpoint
    ↓
Config → Training Script → Multi-GPU → Logging
```

### Evaluation Flow
```
Checkpoint → Model → Sampling → 3DMM → PIRender → Video
    ↓
Config → Evaluation Script → Rendering → Output
```

## 🎯 Key File Relationships

### Configuration Files
- **`config_train.json`** → **`train.py`** → **`models/k_diffusion/config.py`**
- **`config_eval.json`** → **`sample.py`** → **`utils/render.py`**

### Model Files
- **`models/k_diffusion/Diffusion.py`** → **`models/k_diffusion/layers.py`**
- **`models/k_diffusion/sampling.py`** → **`models/k_diffusion/utils.py`**

### Data Files
- **`data/dataset.py`** → **`external/wav2vec2focctc.py`**
- **`data/dataset.py`** → **`external/FaceVerse/`**

### Rendering Files
- **`utils/render.py`** → **`external/PIRender/`**
- **`utils/render.py`** → **`external/FaceVerse/`**

## 🛠️ Development Guidelines

### Adding New Features
1. **Models** - Add to `src/models/k_diffusion/`
2. **Data** - Add to `src/data/`
3. **Utils** - Add to `src/utils/`
4. **Scripts** - Add to `src/scripts/`

### Configuration Management
1. **Training** - Modify `configs/config_train.json`
2. **Evaluation** - Modify `configs/config_eval.json`
3. **Custom** - Create new config based on `configs/config_template.json`

### External Dependencies
1. **Models** - Add to `src/external/`
2. **Checkpoints** - Place in appropriate external subdirectory
3. **Documentation** - Update this file and README.md

## 📝 File Naming Conventions

- **Python files** - `snake_case.py`
- **Configuration files** - `config_<purpose>.json`
- **Shell scripts** - `run_<action>.sh`
- **Documentation** - `UPPER_CASE.md`
- **Directories** - `snake_case/`

## 🔍 Quick Navigation

| Purpose | Location | Key Files |
|---------|----------|-----------|
| **Training** | `src/scripts/` | `train.py`, `run_train.sh` |
| **Evaluation** | `src/scripts/` | `sample.py`, `run_eval.sh` |
| **Model** | `src/models/k_diffusion/` | `Diffusion.py`, `layers.py` |
| **Data** | `src/data/` | `dataset.py` |
| **Rendering** | `src/utils/` | `render.py` |
| **Config** | `configs/` | `config_train.json`, `config_eval.json` |
| **External** | `src/external/` | `wav2vec2focctc.py`, `FaceVerse/`, `PIRender/` |

This structure provides a clean, organized, and scalable foundation for the ReactDiff project, making it easy to understand, maintain, and extend.