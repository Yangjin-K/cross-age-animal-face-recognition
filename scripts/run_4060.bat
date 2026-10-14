@echo off
echo Starting face recognition training with ViT on RTX 4060...
call conda activate pytorch
python code/main.py --config configs/config_4060.yaml
pause