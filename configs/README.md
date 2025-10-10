# ⚙️ ReactDiff Configuration Files

This directory contains configuration files for different phases of the ReactDiff pipeline. Each configuration is optimized for specific use cases and provides fine-grained control over model behavior.

## 📁 Configuration Files

### 🚀 `config_train.json`
**Purpose**: Training configuration  
**Usage**: Used during model training with `run_train.sh` or `train.py`  
**Optimized for**: Fast training with shorter sequences

**Key Settings**:
- **📊 Dataset**: `clip_length: 256` (shorter sequences for training efficiency)
- **🎨 Augmentation**: Enabled (`augment_wrapper: true`) for data diversity
- **📦 Batch Size**: 100 (optimized for training throughput)
- **⚡ Mixed Precision**: FP16 for 2x faster training
- **📈 Logging**: Weights & Biases integration for experiment tracking
- **🔄 Gradient Accumulation**: Configurable for memory optimization

### 🎬 `config_eval.json`
**Purpose**: Evaluation/sampling configuration  
**Usage**: Used during inference with `run_eval.sh` or `sample.py`  
**Optimized for**: High-quality video generation

**Key Settings**:
- **📊 Dataset**: `clip_length: 750` (full 30-second sequences)
- **🎨 Augmentation**: Disabled (`augment_wrapper: false`) for consistent results
- **📦 Batch Size**: 1 (for detailed processing and memory efficiency)
- **🎭 Rendering**: PIRender enabled for realistic video generation
- **⏱️ Window Size**: 64 frames for optimal processing balance
- **🎯 Quality**: High-quality settings for production use

### 📜 `config.json` (Legacy)
**Purpose**: General configuration (kept for backward compatibility)  
**Usage**: Default fallback configuration  
**Note**: Consider using specific configs for better performance

## 🚀 Usage Examples

### 🏋️ Training

#### Single GPU Training
```bash
# Quick start with single GPU
cd src/scripts
./run_train.sh single

# Custom training with specific parameters
python train.py \
    --config ../../configs/config_train.json \
    --out-path ./results/training \
    --name my_model \
    --batch-size 64 \
    --window-size 16
```

#### Multi-GPU Training
```bash
# Distributed training across multiple GPUs
cd src/scripts
./run_train.sh multi

# Custom multi-GPU training
accelerate launch --multi_gpu train.py \
    --config ../../configs/config_train.json \
    --out-path ./results/multi_gpu_training \
    --name my_model
```

### 🎬 Evaluation/Sampling

#### Single GPU Evaluation
```bash
# Quick evaluation with single GPU
cd src/scripts
./run_eval.sh single /path/to/checkpoint.pth

# Custom evaluation with specific settings
python sample.py \
    --config ../../configs/config_eval.json \
    --checkpoint /path/to/checkpoint.pth \
    --out-path ./results/evaluation \
    --window-size 64 \
    --steps 50
```

#### Multi-GPU Evaluation
```bash
# Distributed evaluation across multiple GPUs
cd src/scripts
./run_eval.sh multi /path/to/checkpoint.pth

# Custom multi-GPU evaluation
accelerate launch --multi_gpu sample.py \
    --config ../../configs/config_eval.json \
    --checkpoint /path/to/checkpoint.pth \
    --out-path ./results/multi_gpu_eval
```

### 🧪 Testing Full Sequences
```bash
# Test with full 30-second sequences (recommended)
python test_full_sequence.py \
    --checkpoint /path/to/checkpoint.pth \
    --out-path ./results/full_30s_test

# Custom configuration for testing
python test_full_sequence.py \
    --config configs/config_eval.json \
    --checkpoint /path/to/checkpoint.pth \
    --window-size 64
```

## 🔧 Configuration Parameters

### 🧠 Model Parameters
- **`input_channels`**: 58 (3DMM parameter dimensions: 52 expression + 3 rotation + 3 translation)
- **`depths`**: [2, 4, 4] (U-Net architecture depths for different resolution levels)
- **`channels`**: [128, 256, 512] (Feature channel sizes for each resolution level)
- **`cross_attn_depths`**: [2, 2, 2] (Cross-attention layers for audio-visual fusion)
- **`mapping_out`**: 256 (Output dimension of the mapping network)
- **`dropout_rate`**: 0.05 (Dropout rate for regularization)
- **`sigma_data`**: 0.5 (Data noise level for diffusion)
- **`sigma_min`**: 1e-2 (Minimum noise level)
- **`sigma_max`**: 80 (Maximum noise level)

### 📊 Dataset Parameters
- **`location`**: Path to dataset directory (e.g., "/path/to/REACT2024")
- **`img_size`**: Input image size (256) - Resolution of input images
- **`crop_size`**: Cropped image size (224) - Final image size after cropping
- **`clip_length`**: Sequence length (256 for training, 750 for evaluation)
- **`num_workers`**: Number of data loading workers (4)
- **`pin_memory`**: Pin memory for faster data transfer (true)

### 🏋️ Training Parameters
- **`batch_size`**: Batch size for training (100 for training, 1 for evaluation)
- **`lr`**: Learning rate (1e-4) - Controls training speed
- **`weight_decay`**: L2 regularization (1e-3) - Prevents overfitting
- **`mixed_precision`**: FP16 for faster training and memory efficiency
- **`grad_accum_steps`**: Gradient accumulation steps (1) - For memory optimization
- **`max_epochs`**: Maximum training epochs (1000)
- **`save_every`**: Save checkpoint every N steps (1000)
- **`evaluate_every`**: Evaluate every N steps (100)

### 🎬 Evaluation Parameters
- **`window_size`**: Processing window size (64) - Frames processed at once
- **`steps`**: Sampling steps (50) - Number of denoising steps
- **`momentum`**: Temporal momentum (0.9) - Controls temporal consistency
- **`render_videos`**: Enable video rendering (true) - Generate output videos
- **`save_coeffs`**: Save 3DMM coefficients (true) - Save intermediate results
- **`video_fps`**: Video frame rate (25) - Output video FPS
- **`chunk_size`**: Processing chunk size (32) - Memory optimization

### 🎨 Rendering Parameters
- **`use_pirender`**: Enable PIRender for realistic rendering (true)
- **`save_frames`**: Save individual frames (false)
- **`video_format`**: Output video format ("mp4")
- **`video_codec`**: Video codec ("mp4v")
- **`quality`**: Rendering quality ("high")

## 🛠️ Customization

### 📏 For Different Sequence Lengths
1. **🎬 Short sequences (10s)**: Set `clip_length: 256` (256 frames at 25fps)
2. **🎬 Medium sequences (20s)**: Set `clip_length: 500` (500 frames at 25fps)
3. **🎬 Long sequences (30s)**: Set `clip_length: 750` (750 frames at 25fps)

### 💻 For Different Hardware
1. **💾 Low memory (8GB GPU)**: 
   - Reduce `batch_size` to 16-32
   - Reduce `window_size` to 16-32
   - Enable `grad_accum_steps: 4`
2. **💾 High memory (24GB+ GPU)**:
   - Increase `batch_size` to 64-128
   - Increase `chunk_size` to 64
   - Use larger `window_size: 64`
3. **🖥️ Multi-GPU setup**:
   - Use `run_train.sh multi` or `run_eval.sh multi`
   - Configure `accelerate` for distributed training

### 🎯 For Different Quality
1. **⚡ Fast inference**: 
   - Reduce `steps` to 25
   - Use `window_size: 32`
   - Disable `render_videos` for testing
2. **🎨 High quality**: 
   - Increase `steps` to 100
   - Use `window_size: 64`
   - Enable `quality: "high"`
3. **⚖️ Balanced**: 
   - Use default `steps: 50`
   - Use `window_size: 64`
   - Standard quality settings

## 🐛 Troubleshooting

### Common Issues

#### 1️⃣ Memory Issues
```bash
# Problem: CUDA out of memory
# Solution: Reduce batch size and window size
{
    "training": {
        "batch_size": 16,  # Reduce from 100
        "grad_accum_steps": 4  # Increase from 1
    },
    "evaluation": {
        "window_size": 32,  # Reduce from 64
        "chunk_size": 16    # Reduce from 32
    }
}
```

#### 2️⃣ Performance Issues
```bash
# Problem: Slow training
# Solution: Enable mixed precision and optimize settings
{
    "training": {
        "mixed_precision": "fp16",  # Enable FP16
        "num_workers": 8,           # Increase workers
        "pin_memory": true          # Enable pin memory
    }
}
```

#### 3️⃣ Quality Issues
```bash
# Problem: Poor video quality
# Solution: Increase sampling steps and adjust momentum
{
    "evaluation": {
        "steps": 100,        # Increase from 50
        "momentum": 0.95,    # Increase from 0.9
        "window_size": 64    # Use optimal window size
    }
}
```

#### 4️⃣ Video Length Issues
```bash
# Problem: Short videos (10s instead of 30s)
# Solution: Check clip_length in evaluation config
{
    "dataset": {
        "clip_length": 750  # Should be 750 for 30s at 25fps
    }
}
```

### 🚀 Performance Tips

#### 1️⃣ Training Optimization
- **⚡ Use mixed precision**: `"mixed_precision": "fp16"`
- **📦 Optimize batch size**: Start with 32, adjust based on GPU memory
- **🔄 Use gradient accumulation**: For memory-constrained setups
- **🖥️ Use multi-GPU**: For large-scale training

#### 2️⃣ Inference Optimization
- **⏱️ Optimize window size**: 64 for 30s videos, 16 for 10s videos
- **💾 Use chunked processing**: For long sequences
- **🎯 Balance quality vs speed**: Adjust steps and momentum

#### 3️⃣ Memory Optimization
- **📦 Reduce batch size**: If running out of memory
- **🔄 Use gradient accumulation**: Maintain effective batch size
- **💾 Enable pin memory**: For faster data transfer
- **🎬 Use chunked rendering**: For long video generation

### 🔧 Configuration Validation

```bash
# Validate your configuration
python -c "
import json
import sys

def validate_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check required fields
    required_fields = ['model', 'dataset', 'training']
    for field in required_fields:
        if field not in config:
            print(f'❌ Missing required field: {field}')
            return False
    
    # Check dataset path
    if 'location' not in config['dataset']:
        print('❌ Missing dataset location')
        return False
    
    print(f'✅ Configuration {config_path} is valid!')
    return True

# Validate all configs
configs = ['config_train.json', 'config_eval.json']
for config in configs:
    validate_config(config)
"
```

### 📊 Performance Monitoring

```bash
# Monitor GPU usage during training
watch -n 1 nvidia-smi

# Monitor system resources
htop

# Check training logs
tail -f results/training/logs/training.log
```

---

## 🎯 Best Practices

### 1️⃣ Configuration Management
- **📁 Use separate configs** for training and evaluation
- **🔧 Start with defaults** and adjust based on your needs
- **📝 Document changes** for reproducibility
- **🧪 Test configurations** before long training runs

### 2️⃣ Resource Management
- **💾 Monitor memory usage** during training
- **⚡ Use appropriate batch sizes** for your hardware
- **🔄 Enable gradient accumulation** for memory optimization
- **🖥️ Use multi-GPU** for large-scale training

### 3️⃣ Quality Optimization
- **🎯 Balance quality vs speed** based on your requirements
- **📊 Monitor training metrics** for convergence
- **🎬 Test video generation** regularly
- **🔧 Fine-tune parameters** for your specific use case

---

## 📚 Additional Resources

- **📖 [GETTING_STARTED.md](../GETTING_STARTED.md)** - Step-by-step usage guide
- **🛠️ [ENVIRONMENT_SETUP.md](../ENVIRONMENT_SETUP.md)** - Detailed setup instructions
- **📁 [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md)** - Project organization
- **🐛 [GitHub Issues](https://github.com/your-username/ReactDiff/issues)** - Bug reports and feature requests

Happy configuring! ⚙️✨
