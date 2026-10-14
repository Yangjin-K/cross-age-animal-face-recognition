import torch
import yaml
import sys
import os

# Add the current directory to sys.path so we can import from transfer_learning
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transfer_learning import prepare_evaluate_dataset, extract_features, compute_similarity, evaluate_leave_one_out, CrossAgeFaceModel

def evaluate_models_on_cacd():
    config_path = 'configs/config_progressive.yaml'
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    print("Loading config...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    print("Loading CACD_dataset test data...")
    # prepare_evaluate_dataset will load and filter the dataset accordingly
    # CACD uses 'CACD_dataset' normally in this codebase based on run_progressive_transfer.sh
    _, test_data = prepare_evaluate_dataset('CACD_dataset', config)
    
    if len(test_data) == 0:
        print("Error: Test data is empty. Check dataset loading logic.")
        return

    models_to_test = [
        '../backbone_swin_t_agedb_base.pth',
        '../backbone_swin_t_fgnet_final.pth'
    ]
    
    # Init device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')
    device = torch.device(device)
    
    for model_path in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_path} on CACD_dataset")
        print(f"{'='*60}")
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            continue

        try:
            from transfer_learning import get_id_mapping
            # Not building full model with correct num_classes since we just extract features
            # We use a dummy num_classes=1000 since we just need the backbone
            model = CrossAgeFaceModel(
                embedding_size=config['model'].get('embedding_size', 768),
                num_classes=1000,
                num_periods=10,
                margin=0.1,
                scale=16.0,
                dropout=config['model'].get('dropout', 0.6),
                backbone=config['model'].get('backbone', 'swin_t'),
                orthogonal_strategy=config['model'].get('orthogonal_strategy', 'projection')
            ).to(device)

            print(f"Loading weights from {model_path}...")
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            is_backbone_only = any(k.startswith('features.') for k in state_dict.keys()) and not any(k.startswith('backbone.') for k in state_dict.keys())
            
            if is_backbone_only:
                model.backbone.load_state_dict(state_dict, strict=False)
                print("Loaded backbone-only weights.")
            else:
                keys_to_delete = [k for k in state_dict.keys() if 'arcface_head' in k or 'period_classifier' in k]
                for k in keys_to_delete:
                    del state_dict[k]
                
                model.load_state_dict(state_dict, strict=False)
                print("Loaded full weights (excluding ArcFace/PeriodHeads).")
            
            model.eval()
            print("Extracting features (this may take a while)...")
            features, ids, _ = extract_features(model, test_data, device)
            
            print("Evaluating leave-one-out accuracy...")
            acc = evaluate_leave_one_out(features, ids, device, 'CACD_dataset')
            print(f"[{model_path}] CACD Test Accuracy: {acc:.4f}")
            
        except Exception as e:
            print(f"Error testing {model_path}: {e}")

if __name__ == '__main__':
    # Make sure we run from 'code' directory if paths like '../' are used
    if os.path.basename(os.getcwd()) != 'code':
        os.chdir('code')
    
    evaluate_models_on_cacd()
