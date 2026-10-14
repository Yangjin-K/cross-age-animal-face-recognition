#!/bin/bash

echo "Starting face recognition training with Swin-T (Orthogonal Projection) on RTX 3090..."
# 使用当前环境的 Python
/home/ubuntu/data-disk/envs/face_recog/bin/python code/main.py --config configs/config_swin_t_projection.yaml
read -p "Press Enter to continue..."