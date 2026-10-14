import torch
from torchvision import transforms

# 优化的数据预处理 - 更温和的增强策略
train_transform_optimized = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5), # 水平翻转增强
    transforms.RandomRotation(10), # 减小旋转角度到±10度
    # 移除随机平移和缩放，避免过度变形
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02), # 减小颜色抖动强度
    # 移除随机直方图均衡化，避免过度改变图像特征
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3), # 减小随机擦除概率
    # 移除高斯模糊，避免过度模糊图像
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
