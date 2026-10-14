import torch
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transfer_learning import load_human_face_dataset, split_dataset_by_time, test_pretrained_model

def evaluate_models_on_cacd():
    print("Loading config...")
    with open('configs/config_swin_t_decoupling.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    print("Loading CACD dataset...")
    # Load and split
    data_list = load_human_face_dataset('CACD', config)
    train_data, test_data = split_dataset_by_time(data_list, train_ratio=0.7)
    
    models_to_test = [
        '../backbone_swin_t_agedb_base.pth',
        '../backbone_swin_t_fgnet_final.pth'
    ]
    
    for model_path in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing model: {model_path}")
        print(f"{'='*50}")
        try:
            # test_pretrained_model might expect absolute paths or relative to code/
            test_pretrained_model(train_data, test_data, 'CACD', config, override_pretrained_path=model_path)
        except Exception as e:
            print(f"Error testing {model_path}: {e}")

if __name__ == '__main__':
    evaluate_models_on_cacd()
