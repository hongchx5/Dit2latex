#!/bin/bash
# 推理脚本
# 用法: bash scripts/infer.sh <print_image> <style_image> [output_path]

PRINT_PATH="$1"
STYLE_PATH="$2"
OUTPUT="${3:-output.png}"
CONFIG="${4:-config/default.yaml}"
CKPT="${5:-}"

if [ -z "$PRINT_PATH" ] || [ -z "$STYLE_PATH" ]; then
    echo "Usage: bash scripts/infer.sh <print_image> <style_image> [output_path] [config] [checkpoint]"
    exit 1
fi

CMD="python main.py infer --config $CONFIG --print_path $PRINT_PATH --style_path $STYLE_PATH --output $OUTPUT"

if [ -n "$CKPT" ]; then
    CMD="$CMD --checkpoint $CKPT"
fi

echo "Running: $CMD"
eval $CMD
