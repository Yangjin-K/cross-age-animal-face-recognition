import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from models import create_feature_extractor

# 数据预处理
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_fgnet_dataset():
    """加载FGNET数据集"""
    data_dir = "../HumanfaceDataset/FGNET/images"
    data_list = []
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.lower().endswith('.jpg'):
                try:
                    # 解析文件名：001A11.JPG或001A11.jpg
                    id_str = filename[:3]
                    age = int(filename[4:6])
                    img_path = os.path.join(data_dir, filename)
                    data_list.append((img_path, id_str, age))
                except:
                    continue
    
    print(f"Loaded {len(data_list)} images from FGNET")
    return data_list

def extract_features(model, data_list, device):
    """提取特征"""
    model.eval()
    features = []
    ids = []
    ages = []
    
    with torch.no_grad():
        for img_path, id_str, age in data_list:
            try:
                img = Image.open(img_path).convert('RGB')
                img = test_transform(img).unsqueeze(0).to(device)
                feat = model(img)
                feat = F.normalize(feat, p=2, dim=1)
                features.append(feat.cpu().squeeze())
                ids.append(id_str)
                ages.append(age)
            except:
                continue
    
    features = torch.stack(features)
    return features, ids, ages

def compute_similarity(features):
    """计算相似度矩阵"""
    similarity = torch.matmul(features, features.T)
    return similarity

def evaluate(similarity, ids):
    """评估模型性能"""
    # 获取每个样本最相似的其他样本索引
    top1_indices = similarity.argmax(dim=1)
    
    # 判断rank-1是否正确
    correct = 0
    total = len(ids)
    
    for i, idx in enumerate(top1_indices):
        if ids[i] == ids[idx]:
            correct += 1
    
    accuracy = correct / total
    print(f"Rank-1 Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}')
    device = torch.device(device)
    
    # 创建特征提取器
    model = create_feature_extractor(
        backbone='swin_t',
        embedding_size=768,
        pretrained=False,
        dropout=0.3
    ).to(device)
    
    # 加载预训练权重
    pretrained_path = '../backbone_swin_t_projection_for_inference.pth'
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
        print("Weights loaded successfully")
    else:
        print(f"Error: Pretrained weights not found at {pretrained_path}")
        return
    
    # 加载数据集
    data_list = load_fgnet_dataset()
    
    if len(data_list) < 10:
        print("Insufficient data for testing")
        return
    
    # 提取特征
    print("Extracting features...")
    features, ids, ages = extract_features(model, data_list, device)
    print(f"Extracted features for {len(features)} samples")
    
    # 计算相似度
    print("Computing similarity matrix...")
    similarity = compute_similarity(features)
    
    # 评估
    print("Evaluating...")
    accuracy = evaluate(similarity, ids)
    
    # 检查是否满足要求
    if accuracy >= 0.65:
        print("✅ Model meets the accuracy requirement (≥65%)")
    else:
        print("❌ Model does not meet the accuracy requirement (≥65%)")

if __name__ == "__main__":
    main()