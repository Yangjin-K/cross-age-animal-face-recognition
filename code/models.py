import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# 特征提取器基类
class BaseFeatureExtractor(nn.Module):
    def __init__(self, embedding_size=512, dropout=0.5):
        super().__init__()
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(dropout)

# ResNet50特征提取器
class ResNetFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, embedding_size=512, pretrained=True, dropout=0.5):
        super().__init__(embedding_size, dropout)
        # 加载预训练的 ResNet50
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        # 去掉原始的全连接层和池化层（我们保留全局平均池化）
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # 输出形状: (batch, 2048, 1, 1)
        # 可选：添加一个全连接层将 2048 维映射到 embedding_size
        self.fc = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(dropout),
            nn.Linear(2048, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size)
        )
        # 初始化 fc 层
        nn.init.xavier_normal_(self.fc[2].weight)

    def forward(self, x):
        # x: (batch, 3, 224, 224) 或 (batch, 3, 112, 112) 需要与预训练输入尺寸匹配
        # 如果输入是 112x112，可能需要调整第一层卷积 stride 或 resize 输入，这里假设输入已 resize 到 224
        x = self.features(x)          # (batch, 2048, 1, 1)
        x = x.view(x.size(0), -1)     # (batch, 2048)
        x = self.fc(x)                 # 包含 dropout 和 BN 的 Bottleneck
        return x                       # 未归一化的特征向量

# ViT特征提取器
class ViTFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, embedding_size=512, model_name='vit_b_16', pretrained=True, dropout=0.5):
        super().__init__(embedding_size, dropout)
        # 加载预训练的 ViT
        if model_name == 'vit_b_32':
            vit = models.vit_b_32(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'vit_l_16':
            vit = models.vit_l_16(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            vit = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
        # 移除分类头，只保留特征提取部分
        self.features = nn.Sequential(
            vit.conv_proj,
            vit.encoder
        )
        # 保存class token和位置编码
        self.class_token = vit.class_token
        self.pos_embedding = vit.encoder.pos_embedding
        # 保存原始特征维度
        self.original_embedding_size = vit.heads.head.in_features
        # 替换为 Bottleneck 结构
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.original_embedding_size),
            nn.Dropout(dropout),
            nn.Linear(self.original_embedding_size, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size)
        )
        # 初始化线性层
        nn.init.xavier_normal_(self.fc[2].weight)

    def forward(self, x):
        # x: (batch, 3, 224, 224) 或 (batch, 3, 384, 384) 需要与预训练输入尺寸匹配
        # ViT默认输入尺寸为224x224
        
        # 检查输入尺寸
        if x.dim() != 4 or x.shape[1] != 3:
            print(f"Expected input shape (batch, 3, H, W), got {x.shape}")
            return x
        
        # 处理输入
        x = self.features[0](x)  # patch embedding
        x = x.flatten(2).transpose(1, 2)  # (batch, seq_len, hidden_dim)
        
        # 添加class token和位置编码
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        x = torch.cat([class_token, x], dim=1)  # (batch, seq_len+1, hidden_dim)
        x = x + self.pos_embedding
        
        # 通过Transformer编码器
        x = self.features[1](x)
        
        # 获取CLS token的输出
        x = x[:, 0]  # (batch, hidden_dim)
        
        x = self.fc(x)                 # 包含 dropout 和 BN 的 Bottleneck
        return x                       # 未归一化的特征向量

# SwinTransformer特征提取器
class SwinTransformerFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, embedding_size=512, model_name='swin_t', pretrained=True, dropout=0.5):
        super().__init__(embedding_size, dropout)
        # 加载预训练的Swin Transformer
        if model_name == 'swin_b':
            swin = models.swin_b(weights='IMAGENET1K_V1' if pretrained else None)
        else:  # default to swin_t
            swin = models.swin_t(weights='IMAGENET1K_V1' if pretrained else None)
        # 移除分类头，只保留特征提取部分
        self.features = nn.Sequential(*list(swin.children())[:-1])  # 移除最后的分类头
        # 保存原始特征维度
        self.original_embedding_size = swin.head.in_features
        # 替换为标准的人脸识别 Bottleneck Block
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.original_embedding_size),
            nn.Dropout(dropout),
            nn.Linear(self.original_embedding_size, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size)
        )
        # 初始化线性层
        nn.init.xavier_normal_(self.fc[2].weight)

    def forward(self, x):
        # x: (batch, 3, 224, 224) 需要与预训练输入尺寸匹配
        # Swin Transformer默认输入尺寸为224x224
        
        # 检查输入尺寸
        if x.dim() != 4 or x.shape[1] != 3:
            print(f"Expected input shape (batch, 3, H, W), got {x.shape}")
            return x
        
        # 通过Swin Transformer特征提取
        x = self.features(x)  # 输出形状: (batch, hidden_dim)
        x = x.view(x.size(0), -1)  # 展平
        
        x = self.fc(x)                 # (batch, embedding_size) 走完整个 BN -> Dropout -> Linear -> BN 的 Bottleneck
        return x                       # 未归一化的特征向量

# 特征提取器工厂函数
def create_feature_extractor(backbone='resnet50', embedding_size=512, pretrained=True, dropout=0.5, vit_model_name='vit_b_16', swin_model_name='swin_t'):
    if backbone == 'resnet50':
        return ResNetFeatureExtractor(embedding_size, pretrained, dropout)
    elif backbone == 'vit':
        return ViTFeatureExtractor(embedding_size, vit_model_name, pretrained, dropout)
    elif backbone == 'swin_t' or backbone == 'swin_b':
        return SwinTransformerFeatureExtractor(embedding_size, backbone, pretrained, dropout)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
# 身份分类头
class ArcFaceHead(nn.Module):
    def __init__(self, embedding_size=512, num_classes=1000, margin=0.5, scale=64):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        # 权重矩阵，每一列代表一个类中心
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_normal_(self.weight)

    def forward(self, features, labels):
        """
        features: (batch, embedding_size)  未归一化的特征
        labels: (batch,)  每个样本的真实类别索引
        返回: 用于交叉熵损失的 logits (batch, num_classes)
        """
        # 归一化特征和权重
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)  # (num_classes, embedding_size)

        # 计算余弦相似度矩阵 (batch, num_classes)
        cosine = F.linear(features, weight)  # features @ weight.T

        # 获取每个样本真实类别的余弦值
        batch_size = features.size(0)
        cosine_theta = cosine[torch.arange(batch_size), labels].view(-1, 1)  # (batch, 1)

        # 计算 sin(θ) 用于添加角度 margin
        sin_theta = torch.sqrt(1.0 - torch.pow(cosine_theta, 2))

        # 将 margin 转换为与 features 相同设备和类型的张量
        cos_m = torch.cos(torch.tensor(self.margin, dtype=features.dtype, device=features.device))
        sin_m = torch.sin(torch.tensor(self.margin, dtype=features.dtype, device=features.device))

        # cos(θ + m) = cosθ cos m - sinθ sin m
        cos_theta_m = cosine_theta * cos_m - sin_theta * sin_m

        # 构造新的 logits：真实类别位置放 cos(θ+m)，其余保持 cosθ
        logits = cosine * self.scale  # 先乘以 scale
        logits.scatter_(1, labels.view(-1, 1), cos_theta_m * self.scale)

        return logits
    
# 时期分类器（用于预测采集年份）
class PeriodClassifier(nn.Module):
    def __init__(self, embedding_size=512, num_periods=10):
        super().__init__()
        self.fc1 = nn.Linear(embedding_size, 128)
        self.fc2 = nn.Linear(128, num_periods)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)          # 输出 logits，未经过 softmax
        return x
    
# 梯度反转层
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)

# 训练器
class CrossAgeFaceModel(nn.Module):
    def __init__(self, embedding_size=512, num_classes=1000, num_periods=10,
                 margin=0.5, scale=64, period_loss_weight=0.1, orthogonal_loss_weight=0.1, rev_lambda=0.1, dropout=0.5,
                 backbone='resnet50', vit_model_name='vit_b_16', orthogonal_strategy=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.period_loss_weight = period_loss_weight
        self.orthogonal_loss_weight = orthogonal_loss_weight
        self.rev_lambda = rev_lambda
        self.orthogonal_strategy = orthogonal_strategy  # 可选值: None, 'decoupling', 'projection'

        # 特征提取器
        self.backbone = create_feature_extractor(backbone=backbone, embedding_size=embedding_size, 
                                               pretrained=True, dropout=dropout, 
                                               vit_model_name=vit_model_name)

        # 正交解耦层 (原始方法)
        if self.orthogonal_strategy == 'decoupling':
            # 仿射变换矩阵，用于将特征空间分解为年龄相关和年龄无关的部分
            self.affine_transform = nn.Linear(embedding_size, embedding_size)
            # 初始化仿射变换层
            nn.init.eye_(self.affine_transform.weight)
            nn.init.constant_(self.affine_transform.bias, 0)
        # 正交投影层 (新方法 - 包含非线性激活函数打破线性塌陷)
        elif self.orthogonal_strategy == 'projection':
            # 身份特征提取器 (MLP)
            self.id_feature_extractor = nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.BatchNorm1d(embedding_size),
                nn.PReLU(),
                nn.Linear(embedding_size, embedding_size),
                nn.BatchNorm1d(embedding_size)
            )
            # 年龄特征提取器 (MLP)
            self.age_feature_extractor = nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.BatchNorm1d(embedding_size),
                nn.PReLU(),
                nn.Linear(embedding_size, embedding_size),
                nn.BatchNorm1d(embedding_size)
            )
            # 初始化
            nn.init.xavier_normal_(self.id_feature_extractor[0].weight)
            nn.init.constant_(self.id_feature_extractor[0].bias, 0)
            nn.init.xavier_normal_(self.id_feature_extractor[3].weight)
            nn.init.constant_(self.id_feature_extractor[3].bias, 0)
            nn.init.xavier_normal_(self.age_feature_extractor[0].weight)
            nn.init.constant_(self.age_feature_extractor[0].bias, 0)
            nn.init.xavier_normal_(self.age_feature_extractor[3].weight)
            nn.init.constant_(self.age_feature_extractor[3].bias, 0)

        # ArcFace 头
        self.arcface_head = ArcFaceHead(embedding_size, num_classes, margin, scale)

        # 时期分类器（用于预测采集年份，训练时使用）
        self.period_classifier = PeriodClassifier(embedding_size, num_periods)

    def forward(self, images, labels=None, period_labels=None):
        """
        训练阶段: 返回总损失和各个分量
        推理阶段: 仅返回特征向量
        """
        features = self.backbone(images)  # (batch, embedding_size)

        # 处理正交策略
        if self.orthogonal_strategy == 'decoupling':
            # 正交解耦策略：先应用梯度反转，再使用仿射变换
            features_adv = grad_reverse(features, self.rev_lambda)
            features_transformed = self.affine_transform(features_adv)
            id_features = features  # 身份特征使用原始特征
            age_features = features_transformed  # 年龄特征使用变换后的特征
            loss_orthogonal = torch.tensor(0.0, device=features.device)  # 正交解耦策略不需要正交损失
        elif self.orthogonal_strategy == 'projection':
            # 正交投影策略：分别提取身份和年龄特征
            id_features = self.id_feature_extractor(features)
            age_features = self.age_feature_extractor(features)
            # 计算正交损失：强制身份特征和年龄特征正交
            id_features_norm = F.normalize(id_features, p=2, dim=1)
            age_features_norm = F.normalize(age_features, p=2, dim=1)
            cosine_similarity = (id_features_norm * age_features_norm).sum(dim=1)
            loss_orthogonal = torch.square(cosine_similarity).mean()  # 最小化余弦相似度的平方
        else:
            # 无正交策略
            id_features = features
            age_features = features
            loss_orthogonal = torch.tensor(0.0, device=features.device)

        if labels is not None:  # 训练模式，需要计算损失
            # ArcFace 损失 - 使用身份特征进行身份识别
            logits = self.arcface_head(id_features, labels)
            loss_arcface = F.cross_entropy(logits, labels)

             # 计算准确率
            preds = logits.argmax(dim=1)               # 预测类别 (batch,)
            accuracy = (preds == labels).float().mean() # 批次准确率

            # 时期对抗损失 - 使用年龄特征进行时期预测
            # 对年龄特征不再额外应用梯度反转（因为解耦法已经提前反转过，其他方法在这里反转）
            if self.orthogonal_strategy == 'decoupling':
                period_logits = self.period_classifier(age_features)
            elif self.orthogonal_strategy == 'projection':
                # 必须强制要求 ID 特征作为输入去预测 period 时完全失败（分类交叉熵反转），达到彻底解耦目的
                id_features_adv = grad_reverse(id_features, self.rev_lambda)
                period_logits_id = self.period_classifier(id_features_adv)
                period_logits_age = self.period_classifier(age_features)
                period_logits = period_logits_age # 主支由 age 去预测
            else:
                features_adv = grad_reverse(age_features, self.rev_lambda)
                period_logits = self.period_classifier(features_adv)

            # 处理缺失时期标签
            valid_mask = period_labels != -1
            if valid_mask.any():
                valid_period_labels = period_labels[valid_mask]
                
                # Projection 策略下的双重时期损失约束：age 需要准，id 需要混淆
                if self.orthogonal_strategy == 'projection':
                    loss_period = F.cross_entropy(period_logits_age[valid_mask], valid_period_labels) + \
                                  F.cross_entropy(period_logits_id[valid_mask], valid_period_labels)
                else:
                    valid_period_logits = period_logits[valid_mask]
                    loss_period = F.cross_entropy(valid_period_logits, valid_period_labels)
            else:
                loss_period = torch.tensor(0.0, device=period_logits.device)

            # 总损失
            total_loss = loss_arcface + self.period_loss_weight * loss_period + self.orthogonal_loss_weight * loss_orthogonal

            return total_loss, loss_arcface, loss_period, loss_orthogonal, accuracy
        else:
            # 推理模式：返回归一化的身份特征用于识别
            return F.normalize(id_features, p=2, dim=1)

    def compute_cross_time_contrastive_loss(self, features, labels, period_labels, temperature=0.07):
        """
        计算跨时期对比学习损失
        对于同一个体，拉近其在不同年份的特征向量距离
        对于不同个体，推远距离
        """
        # 归一化特征
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.size(0)
        
        # 计算特征间的余弦相似度
        similarity = torch.matmul(features, features.T)  # (batch, batch)
        
        # 对相似度进行裁剪，避免指数运算溢出
        similarity = torch.clamp(similarity, min=-10.0, max=10.0)
        
        # 创建掩码：同一身份且不同时期
        same_id_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (batch, batch)
        different_period_mask = (period_labels.unsqueeze(0) != period_labels.unsqueeze(1)).float()  # (batch, batch)
        positive_mask = same_id_mask * different_period_mask  # 正样本：同一身份不同时期
        
        # 负样本：不同身份
        negative_mask = 1 - same_id_mask
        
        # 排除自身
        eye_mask = 1 - torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask * eye_mask
        negative_mask = negative_mask * eye_mask
        
        # 计算正样本和负样本的数量
        num_positives = positive_mask.sum(dim=1, keepdim=True)
        num_negatives = negative_mask.sum(dim=1, keepdim=True)
        
        # 避免除零错误
        num_positives = torch.clamp(num_positives, min=1)
        num_negatives = torch.clamp(num_negatives, min=1)
        
        # 计算对比损失
        exp_similarity = torch.exp(similarity / temperature)
        positive_sum = (positive_mask * exp_similarity).sum(dim=1, keepdim=True)
        negative_sum = (negative_mask * exp_similarity).sum(dim=1, keepdim=True)
        
        # 添加epsilon以避免log(0)和除零错误
        epsilon = 1e-10
        loss = -torch.log((positive_sum + epsilon) / (positive_sum + negative_sum + epsilon)).mean()
        
        # 确保loss不是nan或inf
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=features.device)
        
        return loss