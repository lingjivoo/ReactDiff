#!/bin/bash
# ReactDiff Evaluation Script (Single/Multi-GPU)
# Usage: ./run_eval.sh [single|multi] [checkpoint_path]

MODE=${1:-single}
CHECKPOINT=${2:-/path/to/your/checkpoint.pth}

if [ "$MODE" = "multi" ]; then
    echo "Starting multi-GPU sampling..."
    accelerate launch --multi_gpu sample.py \
        --config ../../configs/config_eval.json \
        --checkpoint $CHECKPOINT \
        --out-path ./results/evaluation_multi_gpu \
        --multi-gpu
else
    echo "Starting single-GPU sampling..."
    CUDA_VISIBLE_DEVICES=0 python sample.py \
        --config ../../configs/config_eval.json \
        --checkpoint $CHECKPOINT \
        --out-path ./results/evaluation
fi