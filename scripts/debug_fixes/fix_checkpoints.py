import torch

latest = torch.load('latest_checkpoint.pth', map_location='cpu')
state = latest['model_state_dict'] if 'model_state_dict' in latest else latest

# Find all keys belonging to backbone and feature extractors
inference_state = {}
for k, v in state.items():
    if k.startswith('backbone.'):
        # Remove 'backbone.' prefix so it works as backbone-only OR keep it?
        # In transfer_learning.py:
        # if is_backbone_only: model.backbone.load_state_dict()
        # else: model.load_state_dict()
        inference_state[k] = v
    elif k.startswith('id_feature_extractor.') or k.startswith('age_feature_extractor.'):
        inference_state[k] = v

torch.save(inference_state, 'fixed_swin_t_projection_for_inference.pth')
print("Successfully created fixed_swin_t_projection_for_inference.pth with projection heads!")
