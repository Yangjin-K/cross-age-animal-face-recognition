import re

with open('code/progressive_transfer.py', 'r') as f:
    text = f.read()

# 1. Update argparse in main
parser_old = """    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained weights. Overrides config value.')"""
parser_new = """    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained weights.')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save fine-tuned weights.')
    parser.add_argument('--epochs', type=int, default=20, help='Finetune epochs.')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='LR for backbone.')
    parser.add_argument('--lr_head', type=float, default=1e-3, help='LR for heads.')"""
text = text.replace(parser_old, parser_new)

# 2. Update test_pretrained_model signature in main
call_old = """        acc, tar_at_far_01, rank1 = test_pretrained_model(train_data, test_data, dataset_name, config, args.pretrained_path)"""
call_new = """        acc, tar_at_far_01, rank1 = test_pretrained_model(train_data, test_data, dataset_name, config, args)"""
text = text.replace(call_old, call_new)

# 3. Update test_pretrained_model definition
sig_old = "def test_pretrained_model(train_data, test_data, dataset_name, config, custom_weight_path=None):"
sig_new = "def test_pretrained_model(train_data, test_data, dataset_name, config, args=None):"
text = text.replace(sig_old, sig_new)

# 4. Handle custom_weight_path in test_pretrained_model
weight_path_old = """    if custom_weight_path:
        weight_path = custom_weight_path"""
weight_path_new = """    custom_weight_path = args.pretrained_path if args else None
    if custom_weight_path:
        weight_path = custom_weight_path"""
text = text.replace(weight_path_old, weight_path_new)

# 5. Pass args to finetune
finetune_old = """    model.arcface_head = new_arcface.to(device)
    
    # 微调
    model = finetune_on_target_domain(model, train_data, id_mapping, device, config)"""
finetune_new = """    model.arcface_head = new_arcface.to(device)
    
    # 渐进式微调 (放开Backbone)
    model = finetune_on_target_domain(model, train_data, id_mapping, device, config, args)"""
text = text.replace(finetune_old, finetune_new)

# 6. Rewrite finetune_on_target_domain completely
import_re = re.compile(r"def finetune_on_target_domain\(model, train_data, id_mapping, device, config\):.*?(?=\ndef test_pretrained_model)", re.DOTALL | re.MULTILINE)
finetune_func = """def finetune_on_target_domain(model, train_data, id_mapping, device, config, args=None):
    print(f"Finetuning on target domain ({len(train_data)} samples, {len(id_mapping)} classes)...")
    
    # 分层学习率 (放开 Backbone!)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        param.requires_grad = True # 解冻所有层！
        if name.startswith('backbone.'):
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    lr_bb = args.lr_backbone if args else 1e-5
    lr_hd = args.lr_head if args else 1e-3
    epochs = args.epochs if args else 20
    
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': lr_bb},
        {'params': head_params, 'lr': lr_hd}
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    
    # 强力数据增强（针对小样本过拟合）
    mild_train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = HumanFaceDataset(train_data, transform=mild_train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    model.train()
    for epoch in range(epochs):
        total_loss_sum = 0
        total_acc_sum = 0
        iters = 0
        pbar = tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{epochs}")
        for images, id_strs, ages in pbar:
            images = images.to(device)
            labels = torch.tensor([id_mapping[s] for s in id_strs]).to(device)
            
            optimizer.zero_grad()
            total_loss, loss_arcface, _, _, accuracy = model(images, labels, labels)
            
            total_loss.backward()
            optimizer.step()
            
            total_loss_sum += loss_arcface.item()
            total_acc_sum += accuracy.item()
            iters += 1
            
            pbar.set_postfix({'loss': f"{loss_arcface.item():.4f}", 'acc': f"{accuracy.item():.4f}"})
        
        if scheduler:
            scheduler.step()
    
    if epochs > 0 and iters > 0: 
        print(f"Finetuning completed. Final Avg Loss: {total_loss_sum/iters:.4f}")
        
    # 保存微调后的权重
    if args and args.save_path:
        print(f"Saving progressively finetuned model to {args.save_path}")
        inference_dict = {}
        for k, v in model.state_dict().items():
            if k.startswith('backbone.') or k.startswith('id_feature_extractor.') or k.startswith('age_feature_extractor.'):
                inference_dict[k] = v.cpu()
        torch.save(inference_dict, args.save_path)
            
    return model
"""
text = import_re.sub(finetune_func, text)

with open('code/progressive_transfer.py', 'w') as f:
    f.write(text)
