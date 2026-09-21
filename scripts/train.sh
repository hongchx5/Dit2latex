#!/bin/bash
# 训练脚本
# 用法: bash scripts/train.sh [config_path]

CONFIG="${1:-config/default.yaml}"

echo "Starting training with config: $CONFIG"
python main.py train --config "$CONFIG"
