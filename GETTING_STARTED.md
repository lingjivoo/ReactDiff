# 🚀 Getting Started with ReactDiff

This comprehensive guide will walk you through using ReactDiff from data preparation to generating realistic listener reaction videos.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training Your Model](#training-your-model)
- [Generating Videos](#generating-videos)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## ⚡ Quick Start

### 1️⃣ Prerequisites Check

```bash
# Verify your environment is ready
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')
print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"
```

### 2️⃣ Download Pre-trained Model

```bash
# Download a pre-trained checkpoint (if available)
wget https://your-model-url.com/reactdiff_checkpoint.pth -O checkpoints/reactdiff_pretrained.pth
```

### 3️⃣ Generate Your First Video

```bash
# Quick test with sample data
cd src/scripts
python sample.py \
    --config ../../configs/config_eval.json \
    --checkpoint ../../checkpoints/reactdiff_pretrained.pth \
    --out-path ../../results/quick_test \
    --batch-size 1
```

## 📊 Data Preparation

### 1️⃣ Dataset Structure

Ensure your data follows this structure:

```
data/
├── train/
│   ├── Video_files/
│   │   ├── NoXI/
│   │   └── RECOLA/
│   ├── Audio_files/
│   │   ├── NoXI/
│   │   └── RECOLA/
│   ├── Emotion/
│   │   ├── NoXI/
│   │   └── RECOLA/
│   └── 3D_FV_files/
│       ├── NoXI/
│       └── RECOLA/
├── val/
│   └── [same structure as train]
└── test/
    └── [same structure as train]
```

### 2️⃣ Data Validation

```bash
# Validate your dataset
python -c "
import sys
sys.path.append('src')
from data.dataset import ReactionDataset

# Test dataset loading
dataset = ReactionDataset('data', 'train', load_ref=True)
print(f'✅ Dataset loaded: {len(dataset)} samples')
print('✅ Data validation successful!')
"
```

### 3️⃣ Configuration Update

Update your dataset path in the configuration:

```bash
# Edit evaluation config
nano configs/config_eval.json

# Update dataset location
{
    "dataset": {
        "location": "/path/to/your/data",
        "img_size": 256,
        "crop_size": 224,
        "clip_length": 750
    }
}
```

## 🚀 Training Your Model

### 1️⃣ Single GPU Training

```bash
# Start training with single GPU
cd src/scripts
./run_train.sh single
```

### 2️⃣ Multi-GPU Training

```bash
# Start training with multiple GPUs
cd src/scripts
./run_train.sh multi
```

### 3️⃣ Custom Training

```bash
# Custom training with specific parameters
python train.py \
    --config ../../configs/config_train.json \
    --out-path ../../results/my_training \
    --name my_model \
    --batch-size 64 \
    --window-size 16 \
    --weight-kinematics-loss 0.01 \
    --weight-velocity-loss 1.0 \
    --mixed-precision fp16
```

### 4️⃣ Monitor Training

```bash
# Monitor training progress
# Check logs
tail -f results/my_training/logs/training.log

# Or use Weights & Biases (if configured)
# Visit: https://wandb.ai/your-username/reactdiff
```

## 🎬 Generating Videos

### 1️⃣ Basic Video Generation

```bash
# Generate videos with your trained model
cd src/scripts
python sample.py \
    --config ../../configs/config_eval.json \
    --checkpoint ../../results/my_training/checkpoints/best_model.pth \
    --out-path ../../results/my_videos \
    --batch-size 1
```

### 2️⃣ Full 30-Second Videos

```bash
# Generate complete 30-second videos
python sample.py \
    --config ../../configs/config_eval.json \
    --checkpoint ../../results/my_training/checkpoints/best_model.pth \
    --out-path ../../results/full_videos \
    --window-size 64 \
    --steps 50 \
    --momentum 0.9
```

### 3️⃣ Batch Processing

```bash
# Process multiple videos
python sample.py \
    --config ../../configs/config_eval.json \
    --checkpoint ../../results/my_training/checkpoints/best_model.pth \
    --out-path ../../results/batch_videos \
    --batch-size 4 \
    --n 100
```

## 🔧 Advanced Usage

### 1️⃣ Custom Configuration

Create your own configuration:

```bash
# Copy template
cp configs/config_template.json configs/my_custom_config.json

# Edit configuration
nano configs/my_custom_config.json
```

Example custom configuration:

```json
{
    "model": {
        "input_channels": 58,
        "depths": [2, 4, 4],
        "channels": [128, 256, 512],
        "cross_attn_depths": [2, 2, 2]
    },
    "dataset": {
        "location": "/path/to/your/data",
        "clip_length": 500,
        "img_size": 256,
        "crop_size": 224
    },
    "training": {
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 1e-3
    }
}
```

### 2️⃣ Custom Inference Script

Create a custom inference script:

```python
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from models import k_diffusion as K
from utils.render import Render
from data.dataset import get_dataloader

def custom_inference(config_path, checkpoint_path, output_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inner_model = K.config.make_model(config).eval().to(device)
    inner_model.load_state_dict(torch.load(checkpoint_path)['model_ema'])
    model = K.Denoiser(inner_model, sigma_data=config['model']['sigma_data'])
    
    # Load renderer
    render = Render('cuda', use_pirender=True)
    
    # Your custom inference logic here
    print("Custom inference completed!")

if __name__ == "__main__":
    custom_inference(
        config_path="configs/config_eval.json",
        checkpoint_path="checkpoints/my_model.pth",
        output_path="results/custom_output"
    )
```

### 3️⃣ Multi-GPU Evaluation

```bash
# Use Accelerate for multi-GPU evaluation
accelerate launch --multi_gpu sample.py \
    --config ../../configs/config_eval.json \
    --checkpoint ../../checkpoints/my_model.pth \
    --out-path ../../results/multi_gpu_eval \
    --batch-size 8
```

### 4️⃣ Custom Rendering

```python
# Custom rendering with specific parameters
from utils.render import Render

# Initialize renderer
render = Render('cuda', use_pirender=True)

# Custom rendering parameters
render.rendering(
    output_dir="results/custom",
    video_name="my_video",
    listener_3dmm=your_3dmm_tensor,
    listener_reference=your_reference_image,
    fps=30,  # Custom FPS
    chunk_size=16,  # Custom chunk size
    video_format="mp4",  # Custom format
    quality="high"  # Custom quality
)
```

## 📊 Evaluation and Metrics

### 1️⃣ Quantitative Evaluation

```bash
# Run evaluation metrics
python evaluate_metrics.py \
    --config configs/config_eval.json \
    --checkpoint checkpoints/my_model.pth \
    --out-path results/evaluation \
    --metrics all
```

### 2️⃣ Visual Quality Assessment

```bash
# Generate comparison videos
python compare_videos.py \
    --gt-dir data/test/videos \
    --pred-dir results/my_videos \
    --out-dir results/comparison
```

### 3️⃣ Performance Benchmarking

```bash
# Benchmark inference speed
python benchmark.py \
    --config configs/config_eval.json \
    --checkpoint checkpoints/my_model.pth \
    --batch-sizes 1,2,4,8 \
    --window-sizes 16,32,64
```

## 🐛 Troubleshooting

### Common Issues

#### 1️⃣ Memory Issues

```bash
# Problem: CUDA out of memory
# Solution: Reduce batch size
# Edit config: "batch_size": 16  # Reduce from 32

# Problem: System out of memory
# Solution: Use gradient accumulation
# Edit config: "grad_accum_steps": 4
```

#### 2️⃣ Video Generation Issues

```bash
# Problem: Short videos (10 seconds instead of 30)
# Solution: Check clip_length in config
# Edit config: "clip_length": 750  # Should be 750 for 30s at 25fps

# Problem: PIRender errors
# Solution: Check PIRender checkpoint
ls src/external/PIRender/cur_model_fold.pth
```

#### 3️⃣ Import Errors

```bash
# Problem: ModuleNotFoundError
# Solution: Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Problem: External module not found
# Solution: Check external dependencies
ls src/external/FaceVerse/
ls src/external/PIRender/
```

### Performance Tips

#### 1️⃣ Training Optimization

```bash
# Use mixed precision
# In config: "mixed_precision": "fp16"

# Use gradient accumulation
# In config: "grad_accum_steps": 4

# Use appropriate batch size
# Start with 32, adjust based on GPU memory
```

#### 2️⃣ Inference Optimization

```bash
# Use appropriate window size
# For 30s videos: window_size=64
# For 10s videos: window_size=16

# Use chunked processing
# For long sequences: chunk_size=32
# For short sequences: chunk_size=64
```

## 📚 Next Steps

### 1️⃣ Experiment with Parameters

- Try different window sizes
- Adjust loss weights
- Experiment with different architectures

### 2️⃣ Customize for Your Data

- Adapt to your specific dataset
- Fine-tune hyperparameters
- Add custom preprocessing

### 3️⃣ Extend Functionality

- Add new loss functions
- Implement custom metrics
- Create new rendering methods

### 4️⃣ Contribute

- Submit issues and feature requests
- Contribute code improvements
- Share your results and insights

## 🎯 Example Workflows

### Workflow 1: Quick Demo

```bash
# 1. Download pre-trained model
wget https://your-model-url.com/pretrained.pth

# 2. Generate sample video
python sample.py --config configs/config_eval.json \
    --checkpoint pretrained.pth --out-path demo

# 3. Check results
ls demo/video/
```

### Workflow 2: Full Training

```bash
# 1. Prepare data
python prepare_data.py --data-dir /path/to/data

# 2. Train model
python train.py --config configs/config_train.json \
    --out-path results/training

# 3. Evaluate model
python sample.py --config configs/config_eval.json \
    --checkpoint results/training/best_model.pth \
    --out-path results/evaluation

# 4. Analyze results
python analyze_results.py --results-dir results/evaluation
```

### Workflow 3: Custom Dataset

```bash
# 1. Convert your data to ReactDiff format
python convert_dataset.py --input-dir /path/to/your/data \
    --output-dir data/custom

# 2. Update configuration
cp configs/config_train.json configs/config_custom.json
# Edit config_custom.json with your dataset path

# 3. Train on custom data
python train.py --config configs/config_custom.json \
    --out-path results/custom_training

# 4. Generate videos
python sample.py --config configs/config_eval.json \
    --checkpoint results/custom_training/best_model.pth \
    --out-path results/custom_videos
```

---

## 🎉 Congratulations!

You've successfully set up and used ReactDiff! 

- **📖 Learn more** about the project structure in [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **🔧 Troubleshoot** issues in [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
- **📊 Explore** advanced features in the source code

Happy generating! 🎭✨
