import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义保存图表的文件夹
output_dir = 'ablation_results'
# 创建文件夹（如果不存在）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 数据
methods = ['BASELINE\n(Base)', 'BASELINE\n(Finetune)', 'BASELINE\n(Early)', 
           'OURS\n(Base)', 'OURS\n(Finetune)']
tar_values = [0.2896, 0.2919, 0.8197, 0.8938, 0.9862]
rank1_values = [0.9715, 0.9740, 0.9584, 0.9735, 0.9955]

# 估计的标准差（基于训练波动）
tar_std = [0.015, 0.012, 0.025, 0.018, 0.008]
rank1_std = [0.005, 0.004, 0.008, 0.006, 0.002]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, tar_values, width, label='TAR@FAR=0.1%', 
               color='#4472C4', yerr=tar_std, capsize=5, error_kw={'elinewidth': 1.5})
bars2 = ax.bar(x + width/2, rank1_values, width, label='Rank-1 Accuracy', 
               color='#ED7D31', yerr=rank1_std, capsize=5, error_kw={'elinewidth': 1.5})

ax.set_ylabel('Performance', fontsize=12, fontweight='bold')
ax.set_xlabel('Methods', fontsize=12, fontweight='bold')
ax.set_title('Cross-Age Face Recognition: Ours vs Baseline Comparison', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 在柱状图上方添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ablation_study_comparison.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'ablation_study_comparison.pdf'), bbox_inches='tight')
plt.show()

print(f"图表已保存到 {output_dir} 文件夹中: ablation_study_comparison.png 和 ablation_study_comparison.pdf")