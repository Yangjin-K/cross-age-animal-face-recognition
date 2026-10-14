@echo off
echo Starting face recognition training...
call conda activate pytorch
python code/main.py --config configs/config.yaml
pause
