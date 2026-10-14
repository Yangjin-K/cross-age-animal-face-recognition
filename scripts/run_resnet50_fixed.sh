#!/bin/bash

echo "Starting face recognition training with ResNet50 (Fixed Overfitting) on RTX 3090..."
# 使用当前环境的 Python
/home/ubuntu/data-disk/envs/face_recog/bin/python code/main.py --config configs/config_resnet50_fixed.yaml
read -p "Press Enter to continue..."