#!/bin/bash

echo "Starting face recognition training with Swin-T on RTX 3090..."
# 使用当前环境的 Python
/home/ubuntu/data-disk/envs/face_recog/bin/python code/main.py --config configs/config_3090.yaml
read -p "Press Enter to continue..."
