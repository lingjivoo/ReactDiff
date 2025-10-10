#!/bin/bash

# ReactDiff Installation Script
# This script sets up the ReactDiff environment from scratch

set -e  # Exit on any error

echo "🚀 ReactDiff Environment Setup"
echo "================================"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✅ Conda found. Using conda for environment setup."
    USE_CONDA=true
else
    echo "⚠️  Conda not found. Using pip with virtual environment."
    USE_CONDA=false
fi

# Function to create conda environment
setup_conda() {
    echo "📦 Creating conda environment..."
    conda create -n reactdiff python=3.9 -y
    
    echo "🔄 Activating environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate reactdiff
    
    echo "🔥 Installing PyTorch with CUDA support..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    echo "📚 Installing other dependencies..."
    pip install -r requirements.txt
}

# Function to create virtual environment
setup_venv() {
    echo "📦 Creating virtual environment..."
    python -m venv reactdiff_env
    
    echo "🔄 Activating environment..."
    source reactdiff_env/bin/activate
    
    echo "⬆️  Upgrading pip..."
    pip install --upgrade pip
    
    echo "🔥 Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    echo "📚 Installing other dependencies..."
    pip install -r requirements.txt
}

# Main installation
if [ "$USE_CONDA" = true ]; then
    setup_conda
else
    setup_venv
fi

echo ""
echo "🧪 Verifying installation..."
python -c "
import torch
import torchvision
import transformers
import librosa
import cv2
import numpy as np
import accelerate
import einops
import torchdiffeq
import torchsde
print('✅ All core dependencies imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the environment:"
if [ "$USE_CONDA" = true ]; then
    echo "   conda activate reactdiff"
else
    echo "   source reactdiff_env/bin/activate"
fi
echo "2. Configure your dataset in configs/config.json"
echo "3. Run training: cd src/scripts && ./run_train.sh single"
echo "4. Run sampling: cd src/scripts && ./run_eval.sh single /path/to/checkpoint.pth"
echo ""
echo "📖 For detailed instructions, see ENVIRONMENT_SETUP.md"


