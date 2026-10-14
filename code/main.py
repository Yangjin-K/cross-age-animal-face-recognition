import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

import sys
import time
import pickle
import datetime
import os
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from mydataset import get_dataset,MyDataset
from models import CrossAgeFaceModel

# 尝试导入优化的预处理策略
try:
    from preprocess_optimized import train_transform_optimized, test_transform
    use_optimized_preprocess = True
    print("Using optimized preprocessing strategy")
except ImportError:
    use_optimized_preprocess = False
    print("Using original preprocessing strategy")

# 全局日志文件名，在主程序中初始化
LOG_FILENAME = None

# 配置日志记录器
def log_result(message):
    if LOG_FILENAME is None:
        raise ValueError("LOG_FILENAME has not been initialized")
    with open(LOG_FILENAME, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(message)

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5), # 水平翻转增强
    transforms.RandomRotation(15), # 随机旋转±15度
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # 随机平移和缩放
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # 缓解背景影响
    transforms.RandomEqualize(p=0.2), # 直方图均衡化，应对光照变化
    transforms.ToTensor(),                    
    transforms.RandomErasing(p=0.5), # 模拟遮挡
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # 模拟低画质/失焦
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                    
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def progress_bar(current, total, epoch, totalloss, archeadloss, periodpredictloss, orthogonalloss, acc, width=30): # 进度条
    """current: 当前已处理样本数；total: 总样本数"""
    percent = current / total
    filled = int(width * percent)
    bar = '=' * filled + '-' * (width - filled)
    sys.stdout.write(
        f'\rEpoch {epoch+1:3d} |{bar}| {percent*100:5.1f}% ({current}/{total})'
        f' | totalloss={totalloss:.4f} | archeadloss={archeadloss:.4f} | periodpredictloss={periodpredictloss:.4f} | orthogonalloss={orthogonalloss:.4f} | acc={acc:.4f}'
    )
    sys.stdout.flush()

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, id_str, age = self.data[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = test_transform(img)
        except Exception:
            img = torch.zeros((3,224,224))
        return img, id_str, age

def train(train_dataset, train_loader, test_data, config, stage='pretrain'):
    # 从配置文件读取超参数
    embedding_size = config['model']['embedding_size']
    num_classes = len(train_dataset.id_to_label)   # 训练集中的身份数
    num_periods = 10   # 时期（年份）组数
    backbone = config['model']['backbone']
    # 检查是否有vit配置
    vit_model_name = config['model'].get('vit', {}).get('model_name', 'vit_b_16')
    margin = float(config['model']['arcface']['margin'])
    scale = float(config['model']['arcface']['scale'])
    dropout = float(config['model']['dropout'])
    period_loss_weight = float(config['model']['period_loss_weight'])
    orthogonal_loss_weight = float(config['model'].get('orthogonal_loss_weight', 0.1))
    rev_lambda = float(config['training']['rev_lambda'])
    lr_backbone = float(config['training']['learning_rate']['backbone'])
    lr_head = float(config['training']['learning_rate']['head'])
    weight_decay = float(config['training']['weight_decay'])
    momentum = float(config['training']['momentum'])
    num_epochs = int(config['training']['num_epochs'])
    batch_size = int(config['training']['batch_size'])
    patience = int(config['training'].get('patience', 20))  # 早停 patience

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')
    device = torch.device(device)

    # 读取正交策略配置
    orthogonal_strategy = config.get('model', {}).get('orthogonal_strategy', None)
    
    # 创建模型
    model = CrossAgeFaceModel(
        embedding_size=embedding_size,
        num_classes=num_classes,
        num_periods=num_periods,
        margin=margin,
        scale=scale,
        period_loss_weight=period_loss_weight,
        orthogonal_loss_weight=orthogonal_loss_weight,
        rev_lambda=rev_lambda,
        dropout=dropout,
        backbone=backbone,
        vit_model_name=vit_model_name,
        orthogonal_strategy=orthogonal_strategy
    ).to(device)

    # 如果是finetune阶段，加载预训练权重并冻结部分底层网络
    if stage == 'finetune':
        # 优先读取配置文件中的 pretrained_path，默认使用上一阶段训练产出的 latest_checkpoint.pth
        pretrained_path = config.get('training', {}).get('pretrained_path', '../latest_checkpoint.pth')
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights for finetuning from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=device)
            
            # 兼容包含 loss/epoch 或仅包含模型权重的 checkpoint
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # 判断这是否是一个仅包含 Backbone 权重的 pth (例如以 features. 开头且没挂在 backbone. 命名空间下)
            is_backbone_only = any(k.startswith('features.') for k in state_dict.keys()) and not any(k.startswith('backbone.') for k in state_dict.keys())
            
            if is_backbone_only:
                model.backbone.load_state_dict(state_dict, strict=False)
                print("Detected backbone-only weights. Loaded into Sub-Backbone network successfully.")
            else:
                # 移出分类头（类别数可能不一致），新加的瓶颈层也可以安全略过或利用
                keys_to_delete = [k for k in state_dict.keys() if 'arcface_head' in k or 'period_classifier' in k 
                                  or 'id_feature_extractor' in k or 'age_feature_extractor' in k]
                for k in keys_to_delete:
                    del state_dict[k]
                    
                model.load_state_dict(state_dict, strict=False)
                print("Weights loaded successfully. Classification heads initialized randomly.")
            
            # 冻结底层网络，仅微调高层和分类头
            print("Freezing bottom layers for finetuning")
            # 对于Swin Transformer，我们可以冻结前几层
            if backbone in ['swin_t', 'swin_b']:
                # 冻结前2层
                for i, layer in enumerate(model.backbone.features.children()):
                    if i < 2:
                        for param in layer.parameters():
                            param.requires_grad = False
            # 对于其他backbone，可以根据需要调整冻结策略
            print("Finetuning mode enabled")
        else:
            print(f"Warning: Pretrained weights not found at {pretrained_path}")

    # 优化器：使用AdamW进行更好的泛化和收敛
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
        {'params': model.id_feature_extractor.parameters(), 'lr': lr_head} if hasattr(model, 'id_feature_extractor') else {'params': []},
        {'params': model.age_feature_extractor.parameters(), 'lr': lr_head} if hasattr(model, 'age_feature_extractor') else {'params': []},
        {'params': model.arcface_head.parameters(), 'lr': lr_head},
        {'params': model.period_classifier.parameters(), 'lr': lr_head}
    ], weight_decay=weight_decay)

    # 学习率调度器：带有 Linear Warm-up 的余弦退火，防止 Transformer 早期崩塌
    class WarmupCosineAnnealingLR(optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=1e-7):
            self.warmup_epochs = warmup_epochs
            self.max_epochs = max_epochs
            self.eta_min = eta_min
            super().__init__(optimizer, -1)

        def get_lr(self):
            if self.last_epoch < self.warmup_epochs:
                # 线性 Warmup
                return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
            else:
                # 余弦退火
                progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                import math
                return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]

    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=min(5, num_epochs//10), max_epochs=num_epochs, eta_min=1e-7)

    # 早停相关变量
    best_tar = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # 记录训练过程中的指标
    training_metrics = {
        'epochs': [],
        'train_accuracy': [],
        'train_loss': [],
        'arcface_loss': [],
        'period_loss': [],
        'orthogonal_loss': [],
        'contrastive_loss': [],
        'val_tar': [],
        'learning_rate': []
    }

    model.train()
    # 训练循环
    for epoch in range(num_epochs):
        tic = time.time()

        total_acc = 0.0
        total_acc_count = 0.0
        total_loss = 0.0
        total_arc_loss = 0.0
        total_period_loss = 0.0
        total_orthogonal_loss = 0.0
        total_contrastive_loss = 0.0

        # Dynamic lambda scheduling: start with 0 and gradually increase
        p = epoch / num_epochs
        model.rev_lambda = 0.1 * (2. / (1. + torch.exp(torch.tensor(-10. * p))) - 1.)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True)
        for batch_idx, (images, labels, period_labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            period_labels = period_labels.to(device)

            optimizer.zero_grad()
            # 前向传播获取特征和基本损失
            batch_total_loss, batch_loss_arc, batch_loss_period, batch_loss_orthogonal, accuracy = model(images, labels, period_labels)

            # 计算跨时期对比损失
            features = model.backbone(images)
            batch_contrastive_loss = model.compute_cross_time_contrastive_loss(features, labels, period_labels)

            # 联合优化：ArcFace损失 + 时期对抗损失 + 正交损失 + 跨时期对比损失
            total_batch_loss = batch_total_loss + 0.1 * batch_contrastive_loss
            total_batch_loss.backward()
            optimizer.step()

            total_acc += accuracy
            total_acc_count += 1
            total_loss += total_batch_loss.item()
            total_arc_loss += batch_loss_arc.item()
            total_period_loss += batch_loss_period.item()
            total_orthogonal_loss += batch_loss_orthogonal.item()
            total_contrastive_loss += batch_contrastive_loss.item()

            pbar.set_postfix(
                loss=f"{total_batch_loss.item():.4f}", 
                arc=f"{batch_loss_arc.item():.4f}", 
                orth=f"{batch_loss_orthogonal.item():.4f}",
                acc=f"{accuracy*100:.2f}%"
            )
            
        scheduler.step()
        toc = time.time()
        duration = toc - tic
        acc = total_acc / total_acc_count
        avg_loss = total_loss / total_acc_count
        avg_arc_loss = total_arc_loss / total_acc_count
        avg_period_loss = total_period_loss / total_acc_count
        avg_orthogonal_loss = total_orthogonal_loss / total_acc_count
        avg_contrastive_loss = total_contrastive_loss / total_acc_count
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        log_result(f"\n- Epoch {epoch+1} completed in {duration:.2f}s. acc={acc:.4f}, lambda={model.rev_lambda:.4f}")
        
        # 记录训练指标
        training_metrics['epochs'].append(epoch + 1)
        training_metrics['train_accuracy'].append(acc.item())
        training_metrics['train_loss'].append(avg_loss)
        training_metrics['arcface_loss'].append(avg_arc_loss)
        training_metrics['period_loss'].append(avg_period_loss)
        training_metrics['orthogonal_loss'].append(avg_orthogonal_loss)
        training_metrics['contrastive_loss'].append(avg_contrastive_loss)
        training_metrics['learning_rate'].append(current_lr)

        # 每3个epoch或最后一个epoch进行测试，评估TAR@FAR
        if (epoch + 1) % 3 == 0 or (epoch + 1) == num_epochs:
            # 保存当前模型，包含正交策略信息
            orthogonal_strategy = config.get('model', {}).get('orthogonal_strategy', 'none')
            temp_weight_path = f'backbone_{config["model"]["backbone"]}_{orthogonal_strategy}_for_inference_temp.pth'
            torch.save(model.backbone.state_dict(), temp_weight_path)
            
            # 测试模型
            from models import create_feature_extractor
            # 确保使用与训练时相同的backbone架构
            test_backbone = create_feature_extractor(
                backbone=config['model']['backbone'],
                embedding_size=config['model']['embedding_size'],
                pretrained=False,
                dropout=config['model']['dropout'],
                vit_model_name=vit_model_name
            )
            
            # 加载权重时使用strict=False，避免因架构微小差异导致加载失败
            try:
                # 尝试使用weights_only参数（PyTorch 1.13+）
                try:
                    test_backbone.load_state_dict(torch.load(temp_weight_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True), strict=True)
                except TypeError:
                    # 如果weights_only参数不可用（旧版本PyTorch），则不使用该参数
                    test_backbone.load_state_dict(torch.load(temp_weight_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'), strict=True)
            except RuntimeError as e:
                print(f"Warning: Failed to load weights with strict=True, trying with strict=False: {e}")
                try:
                    # 尝试使用weights_only参数（PyTorch 1.13+）
                    try:
                        test_backbone.load_state_dict(torch.load(temp_weight_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True), strict=False)
                    except TypeError:
                        # 如果weights_only参数不可用（旧版本PyTorch），则不使用该参数
                        test_backbone.load_state_dict(torch.load(temp_weight_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'), strict=False)
                except Exception as e2:
                    print(f"Error loading weights even with strict=False: {e2}")
            test_backbone.eval()
            test_backbone.to(device)
            
            with torch.no_grad():
                # Create a lightweight temporary dataset for test images
                test_dataset = TestDataset(test_data)
                test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

                print(f'\nfeatures calculating start. total {len(test_data)}')
                feat = []
                ids = []
                ages = []

                for imgs, id_strs, batch_ages in test_loader:
                    imgs = imgs.to(device)
                    f = test_backbone(imgs)
                    f = F.normalize(f, p=2, dim=1)
                    feat.append(f.cpu())
                    ids.extend(id_strs)
                    ages.extend(batch_ages)
                    
                    sys.stdout.write(f'\r| Processing {len(ids)}/{len(test_data)} |')
                    sys.stdout.flush()

                feat = torch.cat(feat, dim=0)
                print("\nfeatures calculating complete.")

                print('\npredicting start (chunked cosine similarity).')
                # Chunking computation to prevent OOM
                N = feat.size(0)
                chunk_size = 2000
                # 放弃半精度节省策略，强制使用 float32，避免大量 1.0 的情况摧毁 TAR 数据分布
                sim_matrix = torch.zeros((N, N), dtype=torch.float32) 
                for i in range(0, N, chunk_size):
                    end_i = min(i + chunk_size, N)
                    chunk_feat = feat[i:end_i].to(device)
                    for j in range(0, N, chunk_size):
                        end_j = min(j + chunk_size, N)
                        chunk_feat_2 = feat[j:end_j].to(device)
                        # 全精度计算，禁止出现溢出的 .half() 降级
                        sim_matrix[i:end_i, j:end_j] = (chunk_feat @ chunk_feat_2.T).cpu()

                # 将对角线（自身相似度）设为负无穷，避免自己匹配自己
                sim_matrix.fill_diagonal_(-float('inf'))

                # tar@far
                print('\ntar_at_far calculating start.')
                scores, labels = extract_pairs_fast(sim_matrix, ids)
                tar, actual_far, threshold = compute_tar_at_far_fast(scores, labels, far_target=0.001)
                log_result(f"TAR@FAR=0.1% = {tar:.4f} (actual FAR = {actual_far:.6f}, threshold = {threshold:.4f})")
                
                # 记录验证指标
                training_metrics['val_tar'].append(tar)

                # 早停逻辑
                if tar > best_tar:
                    best_tar = tar
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    # 保存最佳模型，包含正交策略信息
                    orthogonal_strategy = config.get('model', {}).get('orthogonal_strategy', 'none')
                    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    best_weight_path = f'backbone_{config["model"]["backbone"]}_{orthogonal_strategy}_for_inference_{current_time_str}.pth'
                    torch.save(model.backbone.state_dict(), best_weight_path)
                    log_result(f"New best model found at epoch {best_epoch} with TAR@FAR=0.1% = {best_tar:.4f}")
                else:
                    patience_counter += 1
                    log_result(f"No improvement. Patience counter: {patience_counter}/{patience}")
                    
                # 检查是否触发早停
                if patience_counter >= patience:
                    log_result(f"Early stopping triggered at epoch {epoch + 1}. Best epoch: {best_epoch}, Best TAR@FAR=0.1%: {best_tar:.4f}")
                    break

    # 训练结束后，加载最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log_result(f"Loaded best model from epoch {best_epoch}")
    
    checkpoint = {
        'epoch': best_epoch if best_epoch > 0 else num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'period_loss_weight': model.period_loss_weight,  # 若需保存超参数
        'best_tar': best_tar,
        'training_metrics': training_metrics
    }
    torch.save(checkpoint, 'latest_checkpoint.pth')

    # 确保保存最佳模型
    if best_model_state is None:
        # 包含正交策略信息
        orthogonal_strategy = config.get('model', {}).get('orthogonal_strategy', 'none')
        current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        best_weight_path = f'backbone_{config["model"]["backbone"]}_{orthogonal_strategy}_for_inference_{current_time_str}.pth'
        torch.save(model.backbone.state_dict(), best_weight_path)
    
    # 生成并保存训练图表
    generate_training_plots(training_metrics)

def test(test_data, config, path=None, on_origin_device = False, sim_matrix_path=None):
    # 如果没有指定路径，使用包含模型名称和正交策略的默认路径
    if path is None:
        orthogonal_strategy = config.get('model', {}).get('orthogonal_strategy', 'none')
        path = f'backbone_{config["model"]["backbone"]}_{orthogonal_strategy}_for_inference.pth'
    from models import create_feature_extractor
    # 检查是否有vit配置
    vit_model_name = config['model'].get('vit', {}).get('model_name', 'vit_b_16')
    backbone = create_feature_extractor(
        backbone=config['model']['backbone'],
        embedding_size=config['model']['embedding_size'],
        pretrained=False,
        dropout=config['model']['dropout'],
        vit_model_name=vit_model_name
    )

    # 加载权重时使用strict=False，避免因架构微小差异导致加载失败
    try:
        if on_origin_device:
            backbone.load_state_dict(torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True), strict=True)
        else:
            backbone.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    except RuntimeError as e:
        print(f"Warning: Failed to load weights with strict=True, trying with strict=False: {e}")
        if on_origin_device:
            backbone.load_state_dict(torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True), strict=False)
        else:
            backbone.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=False)

    backbone.eval()

    device = torch.device('cuda' if torch.cuda.is_available() and on_origin_device else 'cpu')
    print(f'using {device}')
    backbone.to(device)

    with torch.no_grad():
        if sim_matrix_path is None:
            # Create a lightweight temporary dataset for test images
            test_dataset = TestDataset(test_data)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

            print(f'\nfeatures calculating start. total {len(test_data)}')
            feat = []
            ids = []
            ages = []

            for imgs, id_strs, batch_ages in test_loader:
                imgs = imgs.to(device)
                f = backbone(imgs)
                f = F.normalize(f, p=2, dim=1)
                feat.append(f.cpu())
                ids.extend(id_strs)
                ages.extend(batch_ages)
                
                sys.stdout.write(f'\r| Processing {len(ids)}/{len(test_data)} |')
                sys.stdout.flush()

            feat = torch.cat(feat, dim=0)
            print("\nfeatures calculating complete.")
            
            # 保存特征向量供t-SNE可视化使用
            torch.save(feat, 'features.pt')

            print('\npredicting start (chunked cosine similarity).')
            # Chunking computation to prevent OOM
            N = feat.size(0)
            chunk_size = 2000
            # 放弃半精度节省策略，强制使用 float32，避免大量 1.0 的情况摧毁 TAR 数据分布
            sim_matrix = torch.zeros((N, N), dtype=torch.float32) 
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                chunk_feat = feat[i:end_i].to(device)
                for j in range(0, N, chunk_size):
                    end_j = min(j + chunk_size, N)
                    chunk_feat_2 = feat[j:end_j].to(device)
                    # 全精度计算，禁止出现溢出的 .half() 降级
                    sim_matrix[i:end_i, j:end_j] = (chunk_feat @ chunk_feat_2.T).cpu()

            # 将对角线（自身相似度）设为负无穷，避免自己匹配自己
            sim_matrix.fill_diagonal_(-float('inf'))
            torch.save(sim_matrix, 'sim_matrix.pt')

            with open('test_ids.pkl', 'wb') as f:
                pickle.dump(ids, f)
            with open('test_ages.pkl', 'wb') as f:
                pickle.dump(ages, f)

            print('predicting complete.')

        else:
            sim_matrix = torch.load(sim_matrix_path, weights_only=True)
            with open('test_ids.pkl', 'rb') as f:
                ids = pickle.load(f)
            # 尝试加载ages，如果不存在则创建空列表
            try:
                with open('test_ages.pkl', 'rb') as f:
                    ages = pickle.load(f)
            except FileNotFoundError:
                ages = [0] * len(ids)  # 创建与ids长度相同的默认ages列表

        # rank-1
        print('\nran1_correct calculating start.')
        rank1_correct, true_labels, pred_labels = get_rank1_correct(sim_matrix, ids)
        acc = rank1_correct.mean().item()
        log_result(f'Accuracy / Rank-1 = {acc:2.4f}')
        
        # 保存真实标签和预测标签供后续混淆矩阵使用
        with open('true_labels.pkl', 'wb') as f:
            pickle.dump(true_labels, f)
        with open('pred_labels.pkl', 'wb') as f:
            pickle.dump(pred_labels, f)
        with open('test_ages.pkl', 'wb') as f:
            pickle.dump(ages, f)
        
        # 按年龄组评估性能
        age_group_results = evaluate_by_age_group(sim_matrix, ids, ages)
        log_result("\nPerformance by age group:")
        for age_group, result in age_group_results.items():
            log_result(f"Age group {age_group}: Count={result['count']}, Rank-1 Accuracy={result['rank1_accuracy']:.4f}, TAR@FAR=0.1%={result['tar_at_far']:.4f}")
        
        # 保存年龄组性能结果
        with open('age_group_results.pkl', 'wb') as f:
            pickle.dump(age_group_results, f)
        
        # 绘制混淆矩阵
        current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cm_save_path = f'confusion_matrix_{current_time_str}.png'
        plot_confusion_matrix(true_labels, pred_labels, save_path=cm_save_path)
        log_result(f"Confusion matrix saved to {cm_save_path}")
        
        # 绘制不同年龄组性能对比图
        age_save_path = f'age_group_performance_{current_time_str}.png'
        plot_age_group_performance(age_group_results, save_path=age_save_path)
        log_result(f"Age group performance plot saved to {age_save_path}")
        
        # 绘制t-SNE特征分布可视化
        try:
            if os.path.exists('features.pt'):
                features = torch.load('features.pt')
                tsne_save_path = f'tsne_visualization_{current_time_str}.png'
                plot_tsne_visualization(features, ids, save_path=tsne_save_path)
                log_result(f"t-SNE visualization saved to {tsne_save_path}")
            else:
                log_result("Warning: features.pt not found, skipping t-SNE visualization")
        except Exception as e:
            log_result(f"Error generating t-SNE visualization: {e}")

        # tar@far
        print('\ntar_at_far calculating start.')
        scores, labels = extract_pairs_fast(sim_matrix, ids)
        # 计算单个FAR阈值的TAR
        tar, actual_far, threshold = compute_tar_at_far_fast(scores, labels, far_target=0.001)
        log_result(f"TAR@FAR=0.1% = {tar:.4f} (actual FAR = {actual_far:.6f}, threshold = {threshold:.4f})")
        
        # 计算多个FAR阈值的TAR
        far_targets = [0.0001, 0.001, 0.01, 0.05, 0.1]  # 0.01%, 0.1%, 1%, 5%, 10%
        multi_far_results = compute_tar_at_multiple_far(scores, labels, far_targets)
        log_result("\nTAR@FAR for multiple thresholds:")
        for far_target in far_targets:
            tar, actual_far, threshold = multi_far_results[far_target]
            log_result(f"TAR@FAR={far_target*100:.2f}% = {tar:.4f} (actual FAR = {actual_far:.6f}, threshold = {threshold:.4f})")
        
        # 保存多FAR结果供后续可视化使用
        with open('multi_far_results.pkl', 'wb') as f:
            pickle.dump(multi_far_results, f)
        
        # 计算并绘制ROC曲线
        fpr, tpr = compute_roc_curve(scores, labels)
        if fpr and tpr:
            current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            roc_save_path = f'roc_curve_{current_time_str}.png'
            plot_roc_curve(fpr, tpr, save_path=roc_save_path)
            log_result(f"ROC curve saved to {roc_save_path}")

        print(f'Test complete.')


def get_rank1_correct(sim_matrix, ids):
    # 获取每个样本最相似的其他样本索引
    top1_indices = sim_matrix.argmax(dim=1)  # 形状 (N,)

    # 判断 rank-1 是否正确
    # top1_indices 是整数张量，可以转为列表或直接索引
    top1_indices_list = top1_indices.tolist()  # 或 top1_indices.cpu().numpy()

    # 列表比较
    rank1_correct = [ids[i] == ids[idx] for i, idx in enumerate(top1_indices_list)]

    # 若需要数值形式，可转为 float
    rank1_correct = torch.tensor(rank1_correct, dtype=torch.float)
    # rank1_correct 是一个布尔张量（可转为 float 计算准确率）

    # 返回预测标签和真实标签
    true_labels = ids
    pred_labels = [ids[idx] for idx in top1_indices_list]
    
    return rank1_correct, true_labels, pred_labels

def extract_pairs_fast(sim_matrix, ids):
    N = sim_matrix.shape[0]

    # 将字符串 ids 映射为整数标签
    unique_ids = list(set(ids))
    id_to_int = {id_str: idx for idx, id_str in enumerate(unique_ids)}
    int_ids = torch.tensor([id_to_int[id_str] for id_str in ids], device=sim_matrix.device)

    i, j = torch.triu_indices(N, N, offset=1, device=sim_matrix.device)
    scores = sim_matrix[i, j]
    labels = (int_ids[i] == int_ids[j]).int()   # 1 表示同身份，0 表示不同身份
    return scores, labels

def compute_tar_at_far_fast(scores, labels, far_target=0.001):
    # 转为张量并排序
    scores = torch.as_tensor(scores)
    labels = torch.as_tensor(labels)
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    total_pos = (labels == 1).sum().item()
    total_neg = (labels == 0).sum().item()
    if total_neg == 0:
        return 1.0, 0.0, float('inf')   # 无负样本，FAR 恒为0

    tp = 0
    fp = 0
    best_tar = 0.0
    best_threshold = None
    best_far = 1.0

    i = 0
    n = len(sorted_scores)
    while i < n:
        cur_score = sorted_scores[i]
        # 处理所有分数相同的样本（一次性更新累积值）
        j = i
        while j < n and sorted_scores[j] == cur_score:
            if sorted_labels[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        # 计算当前阈值下的 FAR 和 TAR
        far = fp / total_neg
        tar = tp / total_pos
        if far <= far_target:
            if tar > best_tar:
                best_tar = tar
                best_threshold = cur_score
                best_far = far
        else:
            # FAR 已超目标，后续只会更大，提前终止
            break
        i = j

    return best_tar, best_far, best_threshold

def compute_tar_at_multiple_far(scores, labels, far_targets=[0.0001, 0.001, 0.01, 0.05, 0.1]):
    """计算多个FAR阈值对应的TAR值"""
    # 转为张量并排序
    scores = torch.as_tensor(scores)
    labels = torch.as_tensor(labels)
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    total_pos = (labels == 1).sum().item()
    total_neg = (labels == 0).sum().item()
    if total_neg == 0:
        return {far: (1.0, 0.0, float('inf')) for far in far_targets}

    results = {}
    for far_target in far_targets:
        results[far_target] = (0.0, 0.0, None)

    tp = 0
    fp = 0

    i = 0
    n = len(sorted_scores)
    while i < n:
        cur_score = sorted_scores[i]
        # 处理所有分数相同的样本（一次性更新累积值）
        j = i
        while j < n and sorted_scores[j] == cur_score:
            if sorted_labels[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        # 计算当前阈值下的 FAR 和 TAR
        far = fp / total_neg
        tar = tp / total_pos
        
        # 更新所有尚未达到目标FAR的结果
        for far_target in far_targets:
            if far <= far_target and results[far_target][0] < tar:
                results[far_target] = (tar, far, cur_score)
        
        i = j

    return results

def get_age_group(age):
    """根据年龄返回年龄组"""
    if age < 20:
        return "0-19"
    elif age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    else:
        return "60+"

def evaluate_by_age_group(sim_matrix, ids, ages):
    """按年龄组评估性能"""
    # 首先获取所有年龄组
    age_groups = {get_age_group(age) for age in ages}
    
    # 为每个年龄组存储结果
    age_group_results = {}
    
    for age_group in sorted(age_groups):
        # 找出属于当前年龄组的样本索引
        group_indices = [i for i, age in enumerate(ages) if get_age_group(age) == age_group]
        
        if len(group_indices) < 2:  # 至少需要2个样本才能计算
            age_group_results[age_group] = {
                'count': len(group_indices),
                'rank1_accuracy': 0.0,
                'tar_at_far': 0.0
            }
            continue
        
        # 提取当前年龄组的子相似度矩阵
        group_sim_matrix = sim_matrix[group_indices][:, group_indices]
        group_ids = [ids[i] for i in group_indices]
        
        # 计算Rank-1准确率
        rank1_correct, _, _ = get_rank1_correct(group_sim_matrix, group_ids)
        rank1_accuracy = rank1_correct.mean().item()
        
        # 计算TAR@FAR=0.1%
        scores, labels = extract_pairs_fast(group_sim_matrix, group_ids)
        tar, actual_far, threshold = compute_tar_at_far_fast(scores, labels, far_target=0.001)
        
        age_group_results[age_group] = {
            'count': len(group_indices),
            'rank1_accuracy': rank1_accuracy,
            'tar_at_far': tar
        }
    
    return age_group_results

def compute_roc_curve(scores, labels):
    """计算ROC曲线的FPR和TPR点"""
    # 转为张量并排序
    scores = torch.as_tensor(scores)
    labels = torch.as_tensor(labels)
    sorted_indices = torch.argsort(scores, descending=True)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    total_pos = (labels == 1).sum().item()
    total_neg = (labels == 0).sum().item()
    
    if total_pos == 0 or total_neg == 0:
        return [], []

    tpr = []
    fpr = []
    tp = 0
    fp = 0

    # 添加初始点 (0, 0)
    tpr.append(0.0)
    fpr.append(0.0)

    i = 0
    n = len(sorted_scores)
    while i < n:
        cur_score = sorted_scores[i]
        # 处理所有分数相同的样本（一次性更新累积值）
        j = i
        while j < n and sorted_scores[j] == cur_score:
            if sorted_labels[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        # 计算当前阈值下的 TPR 和 FPR
        current_tpr = tp / total_pos
        current_fpr = fp / total_neg
        tpr.append(current_tpr)
        fpr.append(current_fpr)
        i = j

    # 添加终点 (1, 1)
    tpr.append(1.0)
    fpr.append(1.0)

    return fpr, tpr

def plot_roc_curve(fpr, tpr, save_path='roc_curve.png'):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TAR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")

def plot_confusion_matrix(true_labels, pred_labels, save_path='confusion_matrix.png', max_classes=20):
    """绘制混淆矩阵"""
    # 统计每个类别的出现次数
    from collections import Counter
    label_counts = Counter(true_labels)
    # 获取出现次数最多的max_classes个类别
    top_labels = [label for label, _ in label_counts.most_common(max_classes)]
    
    # 创建标签到索引的映射
    label_to_idx = {label: i for i, label in enumerate(top_labels)}
    
    # 初始化混淆矩阵
    num_classes = len(top_labels)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # 填充混淆矩阵
    for true, pred in zip(true_labels, pred_labels):
        if true in label_to_idx and pred in label_to_idx:
            true_idx = label_to_idx[true]
            pred_idx = label_to_idx[pred]
            conf_matrix[true_idx, pred_idx] += 1
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.colorbar()
    
    # 设置标签
    plt.xticks(np.arange(num_classes), top_labels, rotation=90, fontsize=8)
    plt.yticks(np.arange(num_classes), top_labels, fontsize=8)
    
    # 在矩阵中添加数值
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black' if conf_matrix[i, j] < conf_matrix.max()/2 else 'white')
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Top {max_classes} Classes)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_age_group_performance(age_group_results, save_path='age_group_performance.png'):
    """绘制不同年龄组的性能对比图"""
    age_groups = list(age_group_results.keys())
    rank1_accuracies = [result['rank1_accuracy'] for result in age_group_results.values()]
    tar_at_far = [result['tar_at_far'] for result in age_group_results.values()]
    counts = [result['count'] for result in age_group_results.values()]
    
    plt.figure(figsize=(12, 8))
    
    # 创建两个子图
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    # 绘制Rank-1准确率
    bars1 = ax1.bar(age_groups, rank1_accuracies, color='skyblue')
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Rank-1 Accuracy')
    ax1.set_title('Rank-1 Accuracy by Age Group')
    ax1.set_ylim(0, 1.05)
    
    # 在柱状图上添加数值
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
    
    # 绘制TAR@FAR=0.1%
    bars2 = ax2.bar(age_groups, tar_at_far, color='lightgreen')
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel('TAR@FAR=0.1%')
    ax2.set_title('TAR@FAR=0.1% by Age Group')
    ax2.set_ylim(0, 1.05)
    
    # 在柱状图上添加数值
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Age group performance plot saved to {save_path}")

def plot_tsne_visualization(features, labels, save_path='tsne_visualization.png', n_samples=1000):
    """绘制t-SNE特征分布可视化"""
    from sklearn.manifold import TSNE
    
    # 随机采样，避免样本过多导致计算缓慢
    if len(features) > n_samples:
        indices = np.random.choice(len(features), n_samples, replace=False)
        sampled_features = features[indices]
        sampled_labels = [labels[i] for i in indices]
    else:
        sampled_features = features
        sampled_labels = labels
    
    # 执行t-SNE降维
    print(f"Performing t-SNE on {len(sampled_features)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(sampled_features.cpu().numpy())
    
    # 为不同的标签分配不同的颜色
    unique_labels = list(set(sampled_labels))
    label_to_color = {label: i for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in sampled_labels]
    
    # 绘制t-SNE结果
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE visualization saved to {save_path}")


def generate_training_plots(training_metrics):
    """生成并保存训练过程中的各种图表"""
    # 设置matplotlib字体，避免中文显示问题
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 创建图表保存目录
    plots_dir = 'training_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # 获取当前时间作为文件名前缀
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 训练损失和验证 TAR@FAR 曲线
    plt.figure(figsize=(12, 6))
    
    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(training_metrics['epochs'], training_metrics['train_loss'], label='Total Loss')
    plt.plot(training_metrics['epochs'], training_metrics['arcface_loss'], label='ArcFace Loss')
    plt.plot(training_metrics['epochs'], training_metrics['period_loss'], label='Period Loss')
    plt.plot(training_metrics['epochs'], training_metrics['orthogonal_loss'], label='Orthogonal Loss')
    plt.plot(training_metrics['epochs'], training_metrics['contrastive_loss'], label='Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    # 绘制验证 TAR@FAR
    plt.subplot(1, 2, 2)
    # 由于验证是每3个epoch进行一次，需要调整x轴
    val_epochs = [e for e in training_metrics['epochs'] if e % 3 == 0 or e == training_metrics['epochs'][-1]]
    if len(val_epochs) == len(training_metrics['val_tar']):
        plt.plot(val_epochs, training_metrics['val_tar'], 'o-', label='TAR@FAR=0.1%')
        plt.xlabel('Epoch')
        plt.ylabel('TAR@FAR')
        plt.title('Validation TAR@FAR')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{current_time_str}_loss_tar.png'))
    plt.close()
    
    # 2. 训练准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(training_metrics['epochs'], training_metrics['train_accuracy'], label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{current_time_str}_accuracy.png'))
    plt.close()
    
    # 3. 正交损失单独图表（放大显示）
    plt.figure(figsize=(10, 6))
    plt.plot(training_metrics['epochs'], training_metrics['orthogonal_loss'], label='Orthogonal Loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Orthogonal Loss')
    plt.title('Orthogonal Loss (Magnified)')
    plt.legend()
    plt.grid(True)
    # 自动调整 y 轴范围以显示更多细节
    if len(training_metrics['orthogonal_loss']) > 0:
        min_loss = min(training_metrics['orthogonal_loss'])
        max_loss = max(training_metrics['orthogonal_loss'])
        if max_loss > min_loss:
            plt.ylim([min_loss - 0.01, max_loss + 0.01])
    plt.savefig(os.path.join(plots_dir, f'{current_time_str}_orthogonal_loss.png'))
    plt.close()
    
    # 3. 学习率变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(training_metrics['epochs'], training_metrics['learning_rate'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{current_time_str}_learning_rate.png'))
    plt.close()
    
    # 4. 综合图表
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(training_metrics['epochs'], training_metrics['train_loss'], label='Total Loss')
    plt.plot(training_metrics['epochs'], training_metrics['arcface_loss'], label='ArcFace Loss')
    plt.plot(training_metrics['epochs'], training_metrics['period_loss'], label='Period Loss')
    plt.plot(training_metrics['epochs'], training_metrics['orthogonal_loss'], label='Orthogonal Loss')
    plt.plot(training_metrics['epochs'], training_metrics['contrastive_loss'], label='Contrastive Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(training_metrics['epochs'], training_metrics['train_accuracy'], label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)
    
    # TAR@FAR曲线
    plt.subplot(2, 2, 3)
    if len(val_epochs) == len(training_metrics['val_tar']):
        plt.plot(val_epochs, training_metrics['val_tar'], 'o-', label='TAR@FAR=0.1%')
        plt.xlabel('Epoch')
        plt.ylabel('TAR@FAR')
        plt.title('Validation TAR@FAR')
        plt.legend()
        plt.grid(True)
    
    # 学习率曲线
    plt.subplot(2, 2, 4)
    plt.plot(training_metrics['epochs'], training_metrics['learning_rate'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{current_time_str}_comprehensive.png'))
    plt.close()
    
    log_result(f"Training plots saved to {plots_dir} directory.")

if __name__ == "__main__":
    do_train = True
    do_test = True

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Cross-Age Face Recognition Training')
    parser.add_argument('--config', type=str, default='configs/config_3090.yaml', help='Path to configuration file')
    parser.add_argument('--stage', type=str, default='pretrain', choices=['pretrain', 'finetune'], help='Training stage: pretrain or finetune')
    args = parser.parse_args()

    # 读取配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 读取批次大小
    batch_size = int(config['training']['batch_size'])

    # 创建日志文件夹
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # 初始化生成带时间戳的日志文件名
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILENAME = os.path.join(log_dir, f'training_log_{current_time_str}.txt')

    # 初始化日志文件
    with open(LOG_FILENAME, 'w', encoding='utf-8') as f:
        f.write(f"=== Cross-Age Face Recognition Training Log ({current_time_str}) ===\n")
        f.write(f"Configuration file: {args.config}\n")
        f.write(f"Backbone: {config['model']['backbone']}\n")
        if config['model']['backbone'] == 'vit':
            f.write(f"ViT Model: {config['model']['vit']['model_name']}\n")
        f.write(f"Embedding size: {config['model']['embedding_size']}\n")
        f.write(f"ArcFace margin: {config['model']['arcface']['margin']}\n")
        f.write(f"ArcFace scale: {config['model']['arcface']['scale']}\n")
        f.write(f"Dropout: {config['model']['dropout']}\n")
        f.write(f"Period loss weight: {config['model']['period_loss_weight']}\n")
        f.write(f"Number of epochs: {config['training']['num_epochs']}\n")
        f.write(f"Batch size: {config['training']['batch_size']}\n")
        f.write(f"Backbone learning rate: {config['training']['learning_rate']['backbone']}\n")
        f.write(f"Head learning rate: {config['training']['learning_rate']['head']}\n")
        f.write(f"Weight decay: {config['training']['weight_decay']}\n")
        f.write(f"Momentum: {config['training']['momentum']}\n")
        f.write(f"Scheduler: {config['training']['scheduler']}\n")
        f.write(f"Rev lambda: {config['training']['rev_lambda']}\n")
        f.write(f"Orthogonal strategy: {config.get('model', {}).get('orthogonal_strategy', 'None')}\n")
        f.write(f"Orthogonal loss weight: {config.get('model', {}).get('orthogonal_loss_weight', 0.1)}\n")
        f.write(f"Preload image: {config['data'].get('preload_image', False)}\n")
        f.write(f"Pre dataset: {config['data'].get('pre_dataset', False)}\n")
        f.write("\n")

    log_result("Training script started.")

    tic = time.time()
    # 根据是否启用优化的预处理策略选择变换
    if use_optimized_preprocess:
        transform = train_transform_optimized
    else:
        transform = train_transform
    
    train_dataset, test_data = get_dataset(
        config['data']['csv_path'],
        transform=transform, 
        preload_image=config['data'].get('preload_image', False), 
        pre_dataset=config['data'].get('pre_dataset', False)
    )
    toc = time.time()
    duration = toc - tic
    print(f"data loaded in {duration}s")

    if do_train:
        # 引入并使用 IdentityPeriodSampler 保证 Contrastive Loss 的有效对采样
        from sampler import IdentityPeriodSampler
        num_instances = 4  # 每个人挑选的图片数量（K），保证 K>1 以便组合正负样本对
        # 当 Batch Size 不能整除 4 时，寻找合适的 K
        if batch_size % num_instances != 0:
            num_instances = 2 if batch_size % 2 == 0 else 1
            
        if num_instances > 1:
            print(f"Using PK Sampler for training: P={batch_size//num_instances}, K={num_instances}")
            pk_sampler = IdentityPeriodSampler(train_dataset, batch_size, num_instances)
            train_loader = DataLoader(train_dataset, batch_sampler=pk_sampler, num_workers=4)
        else:
            print("Fallback to normal shuffling")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # 优先读取 yaml 中的 stage 配置，若无则使用命令行的 args.stage
        current_stage = config.get('training', {}).get('stage', args.stage)
        print(f"Starting training with stage: {current_stage}")
        train(train_dataset, train_loader, test_data, config, stage=current_stage)
        del train_loader
    del train_dataset

    if do_test:
        #test(test_data, config, path='backbone_for_inference.pth', on_origin_device=True, sim_matrix_path='sim_matrix.pt')
        test(test_data, config, on_origin_device=True, sim_matrix_path=None)
    del test_data