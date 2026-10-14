@echo off
echo Starting face recognition training with ViT...
call conda activate pytorch
python code/main.py --config configs/config_vit.yaml
pause