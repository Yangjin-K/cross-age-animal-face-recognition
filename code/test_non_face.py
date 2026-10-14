import torch
import torch.nn.functional as F
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

# 设置matplotlib后端，解决图像显示问题
import matplotlib
# 尝试设置交互式后端
try:
    matplotlib.use('TkAgg', force=True)
except Exception:
    try:
        matplotlib.use('Qt5Agg', force=True)
    except Exception:
        matplotlib.use('agg', force=True)  # fallback到非交互式后端

# 设置matplotlib字体，解决字体缺失问题
# 禁用字体警告
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Bitstream Vera Sans', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica', 'Avant Garde', 'sans-serif'],
    'axes.unicode_minus': False,  # 解决负号显示问题
    'text.usetex': False,  # 禁用LaTeX
})
# 禁用matplotlib的字体警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from models import create_feature_extractor
from mydataset import get_dataset

# 数据预处理
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                    
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 全局日志文件名，在主程序中初始化
LOG_FILENAME = None

# 配置日志记录器
def log_result(message):
    if LOG_FILENAME is None:
        raise ValueError("LOG_FILENAME has not been initialized")
    with open(LOG_FILENAME, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(message)

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

def visualize_test_sample(test_data, sim_matrix, ids, index=0, save_path=None):
    """可视化测试样例"""
    # 获取测试样例信息
    img_path, true_id, age = test_data[index]
    
    # 获取最相似的样本索引和相似度
    top1_idx = sim_matrix[index].argmax().item()
    similarity = sim_matrix[index, top1_idx].item()
    pred_id = ids[top1_idx]
    
    # 加载并显示图像
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        img = Image.new('RGB', (224, 224), color='gray')
    
    # 加载预测的图像
    pred_img_path, _, _ = test_data[top1_idx]
    try:
        pred_img = Image.open(pred_img_path).convert('RGB')
    except Exception:
        pred_img = Image.new('RGB', (224, 224), color='gray')
    
    # 绘制可视化结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Test Sample\nTrue ID: {true_id}\nAge: {age}')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(pred_img)
    plt.title(f'Rank-1 Match\nPredicted ID: {pred_id}\nSimilarity: {similarity:.4f}')
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    if not save_path:
        current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'test_sample_visualization_non_face_{current_time_str}.png'
    plt.savefig(save_path)
    print(f"Test sample visualization saved to {save_path}")
    print(f"You can view the image at: {os.path.abspath(save_path)}")
    
    # 直接展示图像
    try:
        plt.show()
        print("Test sample visualization displayed")
    except Exception as e:
        print(f"Could not display image: {e}")
        print("Image has been saved to file instead")
    finally:
        plt.close()

def test(test_data, config, path=None, on_origin_device = False, sim_matrix_path=None):
    # 如果没有指定路径，使用包含模型名称和正交策略的默认路径
    if path is None:
        orthogonal_strategy = config.get('model', {}).get('orthogonal_strategy', 'none')
        path = f'backbone_{config["model"]["backbone"]}_{orthogonal_strategy}_for_inference.pth'
    
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
            torch.save(feat, 'features_non_face.pt')

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
            torch.save(sim_matrix, 'sim_matrix_non_face.pt')

            with open('test_ids_non_face.pkl', 'wb') as f:
                pickle.dump(ids, f)
            with open('test_ages_non_face.pkl', 'wb') as f:
                pickle.dump(ages, f)

            print('predicting complete.')

        else:
            sim_matrix = torch.load(sim_matrix_path, weights_only=True)
            with open('test_ids_non_face.pkl', 'rb') as f:
                ids = pickle.load(f)
            # 尝试加载ages，如果不存在则创建空列表
            try:
                with open('test_ages_non_face.pkl', 'rb') as f:
                    ages = pickle.load(f)
            except FileNotFoundError:
                ages = [0] * len(ids)  # 创建与ids长度相同的默认ages列表

        # rank-1
        print('\nran1_correct calculating start.')
        rank1_correct, true_labels, pred_labels = get_rank1_correct(sim_matrix, ids)
        acc = rank1_correct.mean().item()
        log_result(f'Accuracy / Rank-1 = {acc:2.4f}')
        
        # 保存真实标签和预测标签供后续使用
        with open('true_labels_non_face.pkl', 'wb') as f:
            pickle.dump(true_labels, f)
        with open('pred_labels_non_face.pkl', 'wb') as f:
            pickle.dump(pred_labels, f)
        with open('test_ages_non_face.pkl', 'wb') as f:
            pickle.dump(ages, f)

        # tar@far
        print('\ntar_at_far calculating start.')
        scores, labels = extract_pairs_fast(sim_matrix, ids)
        # 计算单个FAR阈值的TAR
        tar, actual_far, threshold = compute_tar_at_far_fast(scores, labels, far_target=0.001)
        log_result(f"TAR@FAR=0.1% = {tar:.4f} (actual FAR = {actual_far:.6f}, threshold = {threshold:.4f})")
        
        # 保存特征向量供可视化使用
        if sim_matrix_path is None:
            torch.save(feat, 'features_non_face.pt')
        
        # 可视化测试样例（直接show）
        if len(test_data) > 0:
            sample_index = 0  # 选择第一个测试样例
            visualize_test_sample(test_data, sim_matrix, ids, index=sample_index, save_path=None)

        print(f'Test complete.')

def load_non_face_data(config):
    """加载非人脸数据集，使用与main.py相同的方式"""
    # 使用与main.py相同的数据集加载方式
    train_dataset, test_data = get_dataset(
        config['data']['csv_path'],
        transform=test_transform, 
        preload_image=config['data']['preload_image'], 
        pre_dataset=config['data']['pre_dataset']
    )
    return test_data

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Cross-Age Face Recognition Test on Non-Face Dataset')
    parser.add_argument('--config', type=str, default='configs/config_swin_t_projection.yaml', help='Path to configuration file')
    parser.add_argument('--model_path', type=str, default='backbone_swin_t_projection_for_inference.pth', help='Path to pretrained model')
    args = parser.parse_args()

    # 读取配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 创建日志文件夹
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # 初始化生成带时间戳的日志文件名
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILENAME = os.path.join(log_dir, f'test_non_face_log_{current_time_str}.txt')

    # 初始化日志文件
    with open(LOG_FILENAME, 'w', encoding='utf-8') as f:
        f.write(f"=== Cross-Age Face Recognition Non-Face Test Log ({current_time_str}) ===\n")
        f.write(f"Configuration file: {args.config}\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Backbone: {config['model']['backbone']}\n")
        f.write(f"Embedding size: {config['model']['embedding_size']}\n")
        f.write(f"Orthogonal strategy: {config.get('model', {}).get('orthogonal_strategy', 'None')}\n")
        f.write("\n")

    log_result("Non-face test script started.")

    # 加载非人脸数据
    test_data = load_non_face_data(config)
    log_result(f"Loaded {len(test_data)} non-face samples")

    # 运行测试
    test(test_data, config, path=args.model_path, on_origin_device=True, sim_matrix_path=None)
