# Cross-Temporal Primate Face Recognition
## 跨时间域灵长类面部识别系统

> 第十七届中国大学生服务外包创新创业大赛 · A31 题目参赛作品
> 命题方：西北大学 | 完成完整技术方案设计、模型实现与实验评估

## 项目简介

生物面部特征会随时间发生自然变化（发育、衰老、环境适应等），传统识别模型在跨时间识别任务中精度显著下降。本项目针对**非人类生物（灵长类）跨时间域个体识别**这一难题，构建了多任务深度学习框架，并进一步在公开人脸数据集（AgeDB / FG-NET / CACD）上完成迁移泛化验证。

## 任务结构

本项目包含两个层次的实验：

1. **主任务 —— 灵长类跨时间域个体识别**：使用公开灵长类面部数据集，训练-测试时期互斥，测试集包含训练阶段从未出现过的时期。
2. **迁移验证 —— 人类人脸跨年龄识别**：使用公开人脸数据集 AgeDB / FG-NET / CACD，验证模型从灵长类到人类的跨物种泛化能力。

## 核心结果

| 指标         | 数值          | 说明             |
| ------------ | ------------- | ---------------- |
| Rank-1       | **95.69%**    | 跨时期测试集     |
| TAR@FAR=0.1% | **77.94%**    | 安防场景关键指标 |
| 训练规模     | 27,362 张图像 | 多时期面部数据   |
| 训练轮数     | 25 epochs     | 渐进式迁移训练   |

## 技术方案

### 整体架构

多任务学习框架：共享 Swin Transformer 特征提取器，分别输出身份特征与年龄特征，通过 ArcFace 分类头完成身份识别、梯度反转层实现年龄域适应。

### 核心方法

**1. 正交解耦策略（Orthogonal Decoupling）**
将身份特征与年龄特征在特征空间中正交化，减少两者相关性。经对比实验，最终采用 **projection 实现**（实测优于 decoupling 方案）。

**2. 渐进式迁移学习（Progressive Transfer Learning）**
分阶段微调预训练模型，逐步解冻网络层，用于跨域（灵长类 → 人类）适应。

**3. 梯度反转层域适应**
反向传播时反转年龄预测梯度符号，使特征提取器学到年龄无关的身份表示，并通过动态 λ 参数控制反转强度。

**4. 跨时期对比损失**
针对同一身份不同年龄段样本设计对比损失，增强跨年龄识别能力。

**5. 多模型对比**
系统对比 ResNet-50 / ViT / Swin Transformer 三种主干网络。

### 训练策略

- 优化器：AdamW，backbone 2e-4 / head 2e-3 分层学习率
- 调度器：余弦退火 + 线性预热
- 早停：监控 TAR@FAR，patience=15
- 动态梯度反转强度：λ 从 0 逐步升至 0.1

### 数据增强（9 种）

Resize / RandomCrop / 水平翻转 / 随机旋转 / 随机仿射 / 颜色抖动 / 直方图均衡化 / RandomErasing / 高斯模糊 —— 覆盖姿态、光照、遮挡、低画质等多类变异。

## 评估指标

- **Rank-1 准确率**
- **TAR@FAR=0.1%**（题目要求的核心安全指标）
- 年龄段分组性能分析
- PCA 特征可视化 / 混淆矩阵 / ROC 曲线

## 数据集说明

### 灵长类数据集（主任务）

- 类型：公开灵长类（猴科）面部数据集
- 规模：约 27,000 张多时期面部图像
- 训练/测试划分：时期互斥，测试集包含训练未见的时期
- 时间跨度：满足题目要求（≥ 3 年）

### 公开人脸数据集（迁移验证）

- **AgeDB**：https://gitcode.com/open-source-toolkit/82bd6/blob/main/AgeDB.zip
- **FG-NET**：http://yanweifu.github.io/FG_NET_data/FGNET.zip
- **CACD**：https://hyper.ai/cn/datasets/5355

> 因版权与隐私原因，数据集不随代码仓库提供。
> 所用公开数据集版权归各自发布方所有，本项目仅用于学术研究与竞赛。

## 项目结构

    .
    ├── code/
    │   ├── models.py                 # Swin + ArcFace + 年龄预测器 + 梯度反转层
    │   ├── main.py                   # 训练/测试入口
    │   ├── progressive_transfer.py   # 渐进式迁移学习
    │   ├── transfer_learning.py      # 迁移学习
    │   ├── mydataset.py              # 数据集加载
    │   ├── sampler.py                # IdentityPeriodSampler 采样器
    │   ├── preprocess_optimized.py   # 数据预处理
    │   ├── eval_cacd_custom.py       # CACD 数据集评估
    │   ├── eval_zeroshot.py          # 零样本泛化评估
    │   └── test_non_face.py          # 非人脸图像鲁棒性测试
    ├── configs/                      # 15 个模型/训练配置
    ├── scripts/                      # 训练启动脚本（.sh / .bat）
    ├── docs/
    │   ├── figures/                  # 实验结果图
    │   └── logs/                     # 训练与测试日志
    ├── 技术路线文档.md
    ├── 数据集迁移泛化性测试技术路线.md
    ├── 调优建议.md
    └── statement.md

## 复现方法

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据集（自行下载公开数据集）
# AgeDB:  https://gitcode.com/open-source-toolkit/82bd6/blob/main/AgeDB.zip
# FG-NET: http://yanweifu.github.io/FG_NET_data/FGNET.zip
# CACD:   https://hyper.ai/cn/datasets/5355

# 3. 修改 configs/ 中的数据路径

# 4. 训练
python code/main.py --config configs/config_swin_t_decoupling.yaml
```

## 关键文档

- [技术路线文档](技术路线文档.md) —— 完整技术方案（人脸迁移部分）
- [数据集迁移泛化性测试技术路线](数据集迁移泛化性测试技术路线.md)
- [调优建议](调优建议.md)

## 团队说明

本项目为 5 人团队参赛作品，本人全程参与数据处理、模型实现、训练调优与实验评估。

## 致谢

感谢第十七届中国大学生服务外包创新创业大赛组委会与西北大学命题方。

## 许可

本项目仅用于学习与作品展示。所用公开数据集版权归各自发布方所有。