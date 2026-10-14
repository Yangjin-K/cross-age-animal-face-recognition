from transfer_learning import *
import torch
import yaml

with open('configs/config_swin_t_projection.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Mock test for FGNET
device = torch.device('cuda')
model = CrossAgeFaceModel(
    embedding_size=config["model"]["embedding_size"],
    num_classes=1000,
    num_periods=10,
    margin=0.1,
    scale=16.0,
    backbone='swin_t',
    orthogonal_strategy='projection'
).to(device)

state_dict = torch.load('fixed_swin_t_projection_for_inference.pth', map_location=device)
keys_to_delete = [k for k in state_dict.keys() if 'arcface_head' in k or 'period_classifier' in k]
for k in keys_to_delete: del state_dict[k]
model.load_state_dict(state_dict, strict=False)

# Dataset
dataset_name = 'FGNET'
train_data, test_data = prepare_evaluate_dataset(dataset_name, config)

print("Extracting features...")
features, ids, ages = extract_features(model, test_data, device)
id_to_features = {}
for i, id_s in enumerate(ids):
    if id_s not in id_to_features: id_to_features[id_s] = []
    id_to_features[id_s].append((features[i], ages[i]))

multi_image_items = [(id_s, feats_ages) for id_s, feats_ages in id_to_features.items() if len(feats_ages) >= 2]
gallery_features, gallery_ids, probe_features, probe_ids = [], [], [], []

for id_s, feats_ages in multi_image_items:
    feats_ages.sort(key=lambda x: x[1])
    gallery_features.append(feats_ages[0][0])
    gallery_ids.append(id_s)
    for feat, age in feats_ages[1:]:
        probe_features.append(feat)
        probe_ids.append(id_s)

gallery_features = torch.stack(gallery_features).to(device)
probe_features = torch.stack(probe_features).to(device)

sim_matrix = compute_similarity(probe_features, gallery_features)
acc, top1_indices = evaluate_accuracy(sim_matrix, probe_ids, gallery_ids)
print(f"Zero-shot ACC: {acc}")
