import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import sys
import time
import pickle
import datetime
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from models import create_feature_extractor, CrossAgeFaceModel
from mydataset import MyDataset

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class HumanFaceDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, id_str, age = self.data[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception:
            img = torch.zeros((3, 224, 224))
        return img, id_str, age

def load_human_face_dataset(dataset_name, config=None):
    """加载人脸数据集"""
    data_list = []
    
    data_dir = None
    if config and 'data' in config and 'datasets' in config['data'] and dataset_name in config['data']['datasets']:
        data_dir = config['data']['datasets'][dataset_name].get('path', None)
    
    if data_dir is None:
        if dataset_name == "CACD_dataset":
            data_dir = "./HumanfaceDataset/CACD_dataset/CACD/data/CACD2000"
        elif dataset_name == "FGNET":
            data_dir = "./HumanfaceDataset/FGNET/images"
        elif dataset_name == "AgeDB":
            data_dir = "./HumanfaceDataset/AgeDB"
            
    print(f"DEBUG: Selected default data_dir={data_dir}")
    
    import os
    if dataset_name == "CACD_dataset":
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
            import tqdm
            for filename in tqdm.tqdm(files, desc="Loading CACD_dataset"):
                try:
                    parts = filename.split('_')
                    age = int(parts[0])
                    id_str = '_'.join(parts[1:-1])
                    data_list.append((os.path.join(data_dir, filename), id_str, age))
                except:
                    pass
    elif dataset_name == "FGNET":
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.lower().endswith('.jpg')]
            import tqdm
            for filename in tqdm.tqdm(files, desc="Loading FGNET"):
                try:
                    id_str = filename[:3]
                    age = int(filename[4:6])
                    data_list.append((os.path.join(data_dir, filename), id_str, age))
                except:
                    pass
    elif dataset_name == "AgeDB":
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
            import tqdm
            for filename in tqdm.tqdm(files, desc="Loading AgeDB"):
                try:
                    parts = filename.split('_')
                    age = int(parts[-2])
                    id_str = '_'.join(parts[1:-2])
                    data_list.append((os.path.join(data_dir, filename), id_str, age))
                except:
                    pass
                    
    print(f"Loaded {len(data_list)} images from {dataset_name}")
    return data_list

def split_dataset_by_time(data_list, train_ratio=0.7):
    """按时间划分训练集和测试集，确保测试集包含训练集未见过的时期"""
    # 按年龄/年份排序
    data_list.sort(key=lambda x: x[2])
    
    # 计算划分点
    split_point = int(len(data_list) * train_ratio)
    
    # 确保测试集包含训练集未见过的时期
    train_data = data_list[:split_point]
    test_data = data_list[split_point:]
    
    # 检查测试集是否包含新的时期
    train_periods = set([x[2] for x in train_data])
    test_periods = set([x[2] for x in test_data])
    new_periods = test_periods - train_periods
    
    if not new_periods:
        # 如果没有新的时期，调整划分点
        print("Warning: No new periods in test set, adjusting split...")
        # 找到最大的连续时期
        periods = sorted(list(set([x[2] for x in data_list])))
        if len(periods) > 1:
            # 使用最后一个时期作为测试集
            last_period = periods[-1]
            train_data = [x for x in data_list if x[2] != last_period]
            test_data = [x for x in data_list if x[2] == last_period]
    
    print(f"Train set: {len(train_data)} images, Test set: {len(test_data)} images")
    print(f"Train periods: {sorted(set([x[2] for x in train_data]))}")
    print(f"Test periods: {sorted(set([x[2] for x in test_data]))}")
    
    return train_data, test_data

def get_id_mapping(data_list):
    """获取ID到标签的映射"""
    unique_ids = list(set([x[1] for x in data_list]))
    id_to_label = {id_str: idx for idx, id_str in enumerate(unique_ids)}
    return id_to_label

def evaluate_leave_one_out(features, ids, device, dataset_name):
    """对所有 ID 只有一张图片的数据集使用 leave-one-out 评估"""
    print("Evaluating using leave-one-out method...")
    
    # 计算所有样本之间的相似度矩阵
    similarity = torch.matmul(features, features.T)
    
    # 1. 计算 Rank-1 准确率（leave-one-out方式）
    # 对于每个样本，在其他样本中找到最相似的（排除自己）
    similarity_no_diag = similarity.clone()
    for i in range(len(similarity_no_diag)):
        similarity_no_diag[i, i] = -float('inf')
    
    top1_indices = similarity_no_diag.argmax(dim=1)
    
    # 判断rank-1是否正确
    correct = 0
    total = len(ids)
    
    for i, idx in enumerate(top1_indices):
        if ids[i] == ids[idx]:
            correct += 1
    
    rank1_accuracy = correct / total if total > 0 else 0.0
    
    # 2. 计算 ACC（与 Rank-1 相同）
    acc = rank1_accuracy
    
    # 3. 计算 TAR@FAR=0.1
    # 构建正样本对和负样本对的相似度列表
    genuine_scores = []
    impostor_scores = []
    
    n = len(ids)
    total_pairs = n * (n - 1) // 2
    print(f"Calculating similarity pairs for {total_pairs} pairs...")
    
    for i in tqdm(range(n), desc="Calculating similarity pairs"):
        for j in range(i+1, n):
            score = similarity[i, j].item()
            if ids[i] == ids[j]:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)
    
    print(f"Found {len(genuine_scores)} genuine pairs and {len(impostor_scores)} impostor pairs")
    
    # 计算 TAR@FAR=0.1
    tar_at_far_01 = 0.0
    if len(impostor_scores) > 0 and len(genuine_scores) > 0:
        # 对负样本分数排序
        impostor_scores_sorted = sorted(impostor_scores)
        # 找到 FAR=0.1 对应的阈值
        far_01_idx = int(len(impostor_scores) * 0.1)
        if far_01_idx >= len(impostor_scores):
            far_01_idx = len(impostor_scores) - 1
        threshold = impostor_scores_sorted[far_01_idx]
        
        # 计算 TAR（真正例率）
        genuine_accepted = sum(1 for score in genuine_scores if score >= threshold)
        tar_at_far_01 = genuine_accepted / len(genuine_scores)
    elif len(genuine_scores) == 0:
        print(f"Warning: No genuine pairs found in {dataset_name}. Cannot compute TAR@FAR=0.1.")
        tar_at_far_01 = 0.0
    
    # 输出所有指标
    print(f"\n{'='*50}")
    print(f"Evaluation Results for {dataset_name}:")
    print(f"{'='*50}")
    print(f"ACC (Accuracy):          {acc:.4f} ({acc*100:.2f}%)")
    print(f"TAR@FAR=0.1:             {tar_at_far_01:.4f} ({tar_at_far_01*100:.2f}%)")
    print(f"Rank-1 Accuracy:         {rank1_accuracy:.4f} ({rank1_accuracy*100:.2f}%)")
    print(f"{'='*50}\n")
    
    return acc, tar_at_far_01, rank1_accuracy


def finetune_on_target_domain(model, train_data, id_mapping, device, config, args=None):
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
    
    # 根据数据集大小调整数据增强策略
    if args and hasattr(args, 'dataset') and args.dataset == "FGNET":
        # FGNET 数据集较小，使用更激进的数据增强以缓解过拟合
        mild_train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),  # 垂直翻转
            transforms.RandomRotation(30),  # 更大的旋转角度
            transforms.ColorJitter(
                brightness=0.3,  # 更大的亮度变化
                contrast=0.3,    # 更大的对比度变化
                saturation=0.3,  # 更大的饱和度变化
                hue=0.15         # 更大的色调变化
            ),
            transforms.RandomAffine(
                degrees=15,      # 仿射变换角度
                translate=(0.1, 0.1),  # 平移
                scale=(0.8, 1.2),      # 缩放
                shear=10               # 剪切
            ),
            transforms.RandomErasing(p=0.6, scale=(0.02, 0.25)),  # 随机擦除
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # 高斯模糊
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # 其他数据集使用温和的数据增强
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
            period_labels = (torch.ones_like(labels) * -1).to(device)
            total_loss, loss_arcface, _, _, accuracy = model(images, labels, period_labels)
            
            total_loss.backward()
            optimizer.step()
            
            total_loss_sum += loss_arcface.item()
            total_acc_sum += accuracy.item()
            iters += 1
            
            pbar.set_postfix({'loss': f"{loss_arcface.item():.4f}", 'acc': f"{accuracy.item():.4f}"})
        
        if scheduler:
            scheduler.step()
            
        epoch_avg_loss = total_loss_sum / iters if iters > 0 else 0.0
        epoch_avg_acc = total_acc_sum / iters if iters > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} finished, Avg Loss: {epoch_avg_loss:.4f}, Avg Acc: {epoch_avg_acc:.4f}")
        
    # 保存微调后的权重
    if args and args.save_path:
        print(f"Saving progressively finetuned model to {args.save_path}")
        inference_dict = {}
        for k, v in model.state_dict().items():
            if k.startswith('backbone.') or k.startswith('id_feature_extractor.') or k.startswith('age_feature_extractor.'):
                inference_dict[k] = v.cpu()
        torch.save(inference_dict, args.save_path)
            
    return model

def test_pretrained_model(train_data, test_data, dataset_name, config, args=None):
    """使用预训练模型进行测试"""
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')
    device = torch.device(device)
    
    id_mapping = get_id_mapping(train_data)
    num_classes = len(id_mapping)
    print(f"Target domain classes: {num_classes}")

    # 创建具有完整分支投影解耦功能的完整网络进行测试
    model = CrossAgeFaceModel(
        embedding_size=config['model']['embedding_size'],
        num_classes=num_classes,
        num_periods=10,
        margin=0.45,
        scale=64.0,
        period_loss_weight=0.1,
        orthogonal_loss_weight=0.1,
        rev_lambda=0.1,
        dropout=config['model']['dropout'],
        backbone=config['model']['backbone'],
        vit_model_name=config['model'].get('vit', {}).get('model_name', 'vit_b_16'),
        orthogonal_strategy=config.get('model', {}).get('orthogonal_strategy', None)
    ).to(device)
    
    # 从配置文件中获取预训练路径，如果命令行提供了路径则覆盖它
    custom_weight_path = args.pretrained_path if args else None
    if custom_weight_path:
        pretrained_path = custom_weight_path
    else:
        pretrained_path = config.get('training', {}).get('pretrained_path', '../latest_checkpoint.pth')
    
    if os.path.exists(pretrained_path):
        print(f"Loading checkpoint weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # 判断是否只包含了 Backbone
        is_backbone_only = any(k.startswith('features.') for k in state_dict.keys()) and not any(k.startswith('backbone.') for k in state_dict.keys())
        
        if is_backbone_only:
            model.backbone.load_state_dict(state_dict, strict=False)
            print("Detected backbone-only weights. Loaded into Sub-Backbone network successfully.")

        else:
            # 移出全连接分类头的权重，这是因为预训练的 ID 总数和验证集的占位符 1000 不一致
            keys_to_delete = [k for k in state_dict.keys() if 'arcface_head' in k or 'period_classifier' in k]
            for k in keys_to_delete:
                del state_dict[k]
                
            model.load_state_dict(state_dict, strict=False)
            print("Full Weights (excluding ArcFace/PeriodHeads) loaded successfully")
    else:
        print(f"Error: Checkpoint not found at {pretrained_path}")
        return 0.0, 0.0, 0.0
    
    # 5. 微调模型
    print("Starting target domain finetuning...")
    model = finetune_on_target_domain(model, train_data, id_mapping, device, config, args)

    # 提取特征 (仅在 test_data 上提取)
    print("Extracting features on TEST data using DataLoader...")
    features, ids, ages = extract_features(model, test_data, device)
    
    if len(features) < 10:
        print("Insufficient data for testing")
        return 0.0, 0.0, 0.0
    
    # 因为 extract_features 已经返回了处理好的 tensor，不需要再对 list stack
    # 去掉老代码里的 features = torch.stack(features)
    print(f"Extracted features for {len(features)} samples")
    
    # 建立 ID 到 (特征, 年龄) 的映射
    id_to_features = {}
    for i, id_s in enumerate(ids):
        if id_s not in id_to_features:
            id_to_features[id_s] = []
        id_to_features[id_s].append((features[i], ages[i]))
    
    # 检查是否每个 ID 只有一张图片（如 AgeDB）
    single_image_ids = [id_s for id_s, feats_ages in id_to_features.items() if len(feats_ages) < 2]
    multi_image_ids = [id_s for id_s, feats_ages in id_to_features.items() if len(feats_ages) >= 2]
    
    print(f"Total unique IDs: {len(id_to_features)}, Single image IDs: {len(single_image_ids)}, Multi image IDs: {len(multi_image_ids)}")
    
    if len(single_image_ids) == len(id_to_features):
        # 所有 ID 都只有一张图片，使用 leave-one-out 方式
        print(f"All IDs have only 1 image. Using leave-one-out evaluation.")
        return evaluate_leave_one_out(features, ids, device, dataset_name)
    
    gallery_features = []
    gallery_ids = []
    probe_features = []
    probe_ids = []
    
    print(f"Building feature pool for {len(multi_image_ids)} IDs...")
    # 过滤出有多个图片的ID
    multi_image_items = [(id_s, feats_ages) for id_s, feats_ages in id_to_features.items() if len(feats_ages) >= 2]
    
    for id_s, feats_ages in tqdm(multi_image_items, desc="Building feature pool"):
        # 按照年龄升序排序
        feats_ages.sort(key=lambda x: x[1])
        
        # 按照 8:2 的比例划分训练集和测试集
        total_images = len(feats_ages)
        train_size = int(total_images * 0.8)
        
        # 训练集（用于构建特征池）
        gallery_feats_for_id = [f for f, a in feats_ages[:train_size]]
        # 测试集
        probe_feats_for_id = [f for f, a in feats_ages[train_size:]]
        
        if not probe_feats_for_id:
            # 如果测试集为空，调整划分
            if total_images > 1:
                gallery_feats_for_id = [f for f, a in feats_ages[:-1]]
                probe_feats_for_id = [feats_ages[-1][0]]
            else:
                continue
            
        if not gallery_feats_for_id or not probe_feats_for_id:
            continue
            
        # 使用更高级的特征聚合方法，计算所有训练图像的特征并进行加权平均
        # 这样可以包含不同年龄的特征，更具代表性
        features = torch.stack(gallery_feats_for_id)
        
        # 计算特征之间的相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)
        
        # 计算每个特征的中心度（与其他特征的平均相似度）
        centrality = similarity_matrix.mean(dim=1)
        
        # 使用中心度作为权重，加权平均特征
        weights = centrality / centrality.sum()
        avg_feat = torch.sum(features * weights.unsqueeze(1), dim=0)
        
        # L2 归一化作为最终身份特征池
        avg_feat = torch.nn.functional.normalize(avg_feat, p=2, dim=0)
        
        gallery_features.append(avg_feat)
        gallery_ids.append(id_s)
        
        for f in probe_feats_for_id:
            probe_features.append(f)
            probe_ids.append(id_s)
    
    print(f"Built feature pool with {len(gallery_features)} IDs and {len(probe_features)} probes")
            
    if len(probe_features) == 0 or len(gallery_features) == 0:
        print("Empty probe or gallery after splitting by age!")
        return 0.0, 0.0, 0.0

    gallery_features = torch.stack(gallery_features).to(device)
    probe_features = torch.stack(probe_features).to(device)
    
    print(f"Constructed pool with {len(gallery_features)} IDs. Probes: {len(probe_features)}")

    # 计算所有探针与特征池之间的相似度向量运算，快速求取相似度
    print("Computing similarity matrix...")
    similarity = torch.matmul(probe_features, gallery_features.T)
    
    print("Evaluating...")
    
    # 1. & 2. 计算 Rank-1 Accuracy / ACC
    print(f"Evaluating {len(probe_ids)} probes...")
    top1_indices = similarity.argmax(dim=1).cpu().numpy()
    
    correct = 0
    for i, idx in tqdm(enumerate(top1_indices), desc="Evaluating probes", total=len(top1_indices)):
        if probe_ids[i] == gallery_ids[idx]:
            correct += 1
    
    rank1_accuracy = correct / len(probe_ids) if len(probe_ids) > 0 else 0.0
    acc = rank1_accuracy
    print(f"Current ACC: {acc:.4f} ({acc*100:.2f}%)")

    # 3. 计算 TAR@FAR=0.1
    probe_ids_arr = np.array(probe_ids)
    gallery_ids_arr = np.array(gallery_ids)
    genuine_mask = (probe_ids_arr[:, None] == gallery_ids_arr[None, :])
    impostor_mask = ~genuine_mask
    
    sim_cpu = similarity.cpu().numpy()
    genuine_scores = sim_cpu[genuine_mask]
    impostor_scores = sim_cpu[impostor_mask]
    
    tar_at_far_01 = 0.0
    if len(impostor_scores) > 0 and len(genuine_scores) > 0:
        # np.percentile 可以用 C 底层算法快速找到分位点记录TAR
        threshold = np.percentile(impostor_scores, 90)  # FAR=0.1 即排在 90% 的分位数作为阈值
        genuine_accepted = np.sum(genuine_scores >= threshold)
        tar_at_far_01 = genuine_accepted / len(genuine_scores)
    
    # 输出所有指标
    print(f"\n{'='*50}")
    print(f"Evaluation Results for {dataset_name}:")
    print(f"{'='*50}")
    print(f"ACC (Accuracy):          {acc:.4f} ({acc*100:.2f}%)")
    print(f"TAR@FAR=0.1:             {tar_at_far_01:.4f} ({tar_at_far_01*100:.2f}%)")
    print(f"Rank-1 Accuracy:         {rank1_accuracy:.4f} ({rank1_accuracy*100:.2f}%)")
    print(f"{'='*50}\n")
    
    return acc, tar_at_far_01, rank1_accuracy

def extract_features(model, data_list, device):
    """提取特征用于计算相似度"""
    model.eval()
    features = []
    ids = []
    ages = []
    
    test_dataset = HumanFaceDataset(data_list, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    print(f"Extracting features for {len(test_dataset)} samples...")
    with torch.no_grad():
        for images, id_strs, batch_ages in tqdm(test_loader, desc="Extracting features"):
            images = images.to(device)
            # 直接调用 model(images)，这会触发 CrossAgeFaceModel 里的 id_feature_extractor 并在推理模式下自动归一化
            feat = model(images)
            features.append(feat.cpu())
            ids.extend(id_strs)
            ages.extend(batch_ages)
    
    features = torch.cat(features, dim=0)
    print(f"Extracted features shape: {features.shape}")
    return features, ids, ages

def compute_accuracy(sim_matrix, ids):
    """计算准确率"""
    # 获取每个样本最相似的其他样本索引
    top1_indices = sim_matrix.argmax(dim=1)
    
    # 判断rank-1是否正确
    rank1_correct = [ids[i] == ids[idx] for i, idx in enumerate(top1_indices.tolist())]
    rank1_correct = torch.tensor(rank1_correct, dtype=torch.float)
    
    return rank1_correct.mean().item()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Transfer Learning for Human Face Datasets')
    parser.add_argument('--config', type=str, default='configs/config_3090.yaml', help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='all', choices=['FGNET', 'AgeDB', 'CACD_dataset', 'all'], help='Dataset to test')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained weights.')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save fine-tuned weights.')
    parser.add_argument('--epochs', type=int, default=20, help='Finetune epochs.')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='LR for backbone.')
    parser.add_argument('--lr_head', type=float, default=1e-3, help='LR for heads.')
    args = parser.parse_args()

    # Setup logger
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"progressive_transfer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    
    # Overwrite print with logging
    global print
    def print(*args, **kwargs):
        msg = " ".join(map(str, args))
        logging.info(msg)

    
    # 读取配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 从命令行参数或配置文件中获取数据集列表
    if args.dataset == 'all':
        # 测试所有数据集
        datasets = ["FGNET", "AgeDB", "CACD_dataset"]
    else:
        # 测试指定的单个数据集
        datasets = [args.dataset]
    
    # 结果字典 - 存储三个指标
    results = {}
    
    for dataset_name in datasets:
        print(f"\n=== Processing {dataset_name} ===")
        
        # 加载数据集
        data_list = load_human_face_dataset(dataset_name, config)
        
        if not data_list:
            print(f"No data found for {dataset_name}")
            continue
        
        if len(data_list) < 10:
            print(f"Insufficient data for {dataset_name}")
            continue
        
        # 划分数据集 (按时间划分，无重叠)
        train_data, test_data = split_dataset_by_time(data_list, train_ratio=0.7)
        if len(train_data) < 10 or len(test_data) < 10:
            print(f"Insufficient data after split for {dataset_name}. Skipping...")
            continue
            
        # 使用预训练模型进行测试
        # 传递 args 以允许覆盖 pretrained_path
        acc, tar_at_far_01, rank1 = test_pretrained_model(train_data, test_data, dataset_name, config, args)
        results[dataset_name] = {
            'acc': acc,
            'tar_at_far_01': tar_at_far_01,
            'rank1': rank1
        }
    
    # 打印最终结果汇总
    print("\n" + "="*70)
    print("TRANSFER LEARNING RESULTS SUMMARY")
    print("="*70)
    print(f"{'Dataset':<20} {'ACC':<15} {'TAR@FAR=0.1':<15} {'Rank-1':<15}")
    print("-"*70)
    for dataset_name, metrics in results.items():
        print(f"{dataset_name:<20} {metrics['acc']:<15.4f} {metrics['tar_at_far_01']:<15.4f} {metrics['rank1']:<15.4f}")
    print("="*70)
    
    # 检查是否满足要求
    print("\n=== Requirement Check ===")
    all_above_65 = all(metrics['rank1'] >= 0.65 for metrics in results.values())
    if all_above_65:
        print("✅ All datasets meet the accuracy requirement (Rank-1 ≥ 65%)")
    else:
        print("❌ Some datasets do not meet the accuracy requirement (Rank-1 ≥ 65%)")
        for dataset_name, metrics in results.items():
            if metrics['rank1'] < 0.65:
                print(f"{dataset_name}: Rank-1 = {metrics['rank1']:.4f} < 65%")

if __name__ == "__main__":
    main()