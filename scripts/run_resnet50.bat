@echo off
echo Starting face recognition training with ResNet50...
call conda activate pytorch
python code/main.py --config configs/config_resnet50.yaml
pause