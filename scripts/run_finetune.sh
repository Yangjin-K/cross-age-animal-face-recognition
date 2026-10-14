#!/bin/bash

# ========================================================
# Cross-Age Face Recognition: Finetuning Script
# 基于 Swin-T 的人脸特定微调重启脚本
# ========================================================

# 1. 切换到项目根目录，以保证正确的基于根目录的相对路径读取（比如 ./data）
cd "$(dirname "$0")/.." || exit

# 2. 指定使用3号显卡
export CUDA_VISIBLE_DEVICES=3

# 3. 打印启动日志
echo "Starting Face Domain Finetuning with PK Sampler..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Config file: configs/config_finetune.yaml"

# 4. 执行微调主程序 (在根目录下调用 code/main.py)
python code/main.py --config configs/config_finetune.yaml

echo "Finetuning process exited."
