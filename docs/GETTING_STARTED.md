# Getting Started with ReactDiff

This guide will help you get up and running with ReactDiff quickly.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- At least 8GB RAM
- 20GB+ free disk space

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ReactDiff
```

### 2. Create Virtual Environment

```bash
python -m venv reactdiff_env
source reactdiff_env/bin/activate  # On Windows: reactdiff_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download External Dependencies

The project requires some external models and data files:

```bash
# Download Wav2Vec2 model (if not already present)
# This should be in external/facebook/wav2vec2-base-960h/

# Download FaceVerse components (if not already present)
# This should be in external/FaceVerse/

# Download reference face data
# This should be in external/reference_full.npy
```

## Quick Test

### 1. Prepare a Small Dataset

Create a minimal test dataset with the following structure:
```
test_data/
├── Audio_files/
│   └── sample.wav
├── SL-crop/Video_files/
│   └── sample.mp4
├── Emotion/
│   └── sample.csv
├── 3D_FV_files/
│   └── sample.npy
└── paper-list/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

### 2. Update Configuration

Edit `configs/config.json` to point to your test dataset:
```json
{
    "dataset": {
        "location": "/path/to/your/test_data",
        "img_size": 256,
        "crop_size": 224,
        "clip_length": 256
    }
}
```

### 3. Run a Quick Training Test

```bash
cd src/scripts
python train.py \
    --config ../../configs/config.json \
    --out-path ./test_results \
    --name test_model \
    --batch-size 2 \
    --evaluate-every 100 \
    --save-every 100
```

### 4. Test Sampling

```bash
cd src/scripts
python sample.py \
    --config ../../configs/config.json \
    --checkpoint ./test_results/test_model_00000100.pth \
    --out-path ./test_samples \
    --batch-size 1
```

## Understanding the Output

### Training Output

- **Checkpoints**: Saved in `--out-path` directory
- **Logs**: Training progress and loss values
- **Evaluation**: Periodic evaluation during training

### Sampling Output

- **Videos**: Rendered listener reaction videos
- **3DMM Parameters**: Raw generated parameters
- **Metrics**: Evaluation metrics if ground truth available

## Common Issues and Solutions

### Issue: "ModuleNotFoundError"

**Solution**: Make sure you're running scripts from the correct directory and all paths are updated.

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size in your configuration or script arguments.

### Issue: "File not found" errors

**Solution**: Check that all external dependencies are properly downloaded and paths are correct.

### Issue: Import errors

**Solution**: Ensure you've installed all dependencies and the virtual environment is activated.

## Next Steps

1. **Explore the Configuration**: Understand the model parameters in `configs/config.json`
2. **Read the Code**: Start with `src/data/dataset.py` to understand data loading
3. **Experiment**: Try different hyperparameters and model configurations
4. **Evaluate**: Use the provided metrics to assess model performance

## Getting Help

- Check the main README.md for detailed documentation
- Review the code comments for implementation details
- Look at the example scripts in `examples/` directory
