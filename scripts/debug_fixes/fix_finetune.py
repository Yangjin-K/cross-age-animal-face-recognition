with open("code/transfer_learning.py", "r") as f:
    text = f.read()

import re

# Update learning rate and weight decay
text = text.replace("    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=5e-4)", 
                    "    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3, weight_decay=1e-4)\n    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)")

# Update transformation to a mildly augmented one inside finetune_on_target_domain
text = re.sub(r'train_dataset = HumanFaceDataset\(train_data, transform=train_transform\)',
              """    # 使用温和的增强策略加速域自适应收敛
    mild_train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = HumanFaceDataset(train_data, transform=mild_train_transform)""", text)

# Add scheduler.step()
text = text.replace("            total_loss_sum += total_loss.item()", "            total_loss_sum += total_loss.item()")
text = text.replace("pbar.set_postfix({'loss': f'{total_loss.item():.4f}', 'acc': f'{accuracy.item():.4f}'})",
                    "pbar.set_postfix({'loss': f'{total_loss.item():.4f}', 'acc': f'{accuracy.item():.4f}'})\n        scheduler.step()")


with open("code/transfer_learning.py", "w") as f:
    f.write(text)
