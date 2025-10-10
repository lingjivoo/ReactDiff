#!/bin/bash
# ReactDiff Training Script (Single/Multi-GPU)
# Usage: ./run_train.sh [single|multi]

MODE=${1:-single}

if [ "$MODE" = "multi" ]; then
    echo "Starting multi-GPU training..."
    accelerate launch --multi_gpu train.py \
        --config ../../configs/config_train.json \
        --out-path ./results/training_multi_gpu \
        --name reactdiff_multi_gpu \
        --multi-gpu
else
    echo "Starting single-GPU training..."
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --config ../../configs/config_train.json \
        --out-path ./results/training \
        --name reactdiff_model
fi
