"""快速绘制 recall 曲线"""
import matplotlib.pyplot as plt
import os
import sys

# 添加项目根目录到路径
__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(__FILE_DIR__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path

# 设置缓存路径
__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(project_root, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)

FONTSIZE = 18

def convert_to_list(data):
    """将 CUDA tensor 或 numpy 数组转换为 Python 列表"""
    import torch
    import numpy as np
    
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.detach().cpu().numpy()
        else:
            data = data.detach().numpy()
    
    if isinstance(data, np.ndarray):
        data = data.tolist()
    
    if isinstance(data, list):
        data = [float(item.cpu().item() if isinstance(item, torch.Tensor) and item.is_cuda 
                     else item.item() if isinstance(item, torch.Tensor) 
                     else float(item)) for item in data]
    
    return data

# 加载记录
title = 'CMomentum_label_flipping_HSM-Adaptive'
path_list = ['NeuralNetwork_cifar10', 'Centralized_n=10_b=3', 'iidPartition']

try:
    record = load_file_in_cache(title, path_list)
    print(f"成功加载记录: {title}")
except Exception as e:
    print(f"无法加载记录: {e}")
    sys.exit(1)

if 'detection_recall_path' not in record:
    print("警告: 记录文件中没有 detection_recall_path 数据")
    sys.exit(1)

# 提取数据
recall_path = convert_to_list(record.get('detection_recall_path', []))
precision_path = convert_to_list(record.get('detection_precision_path', []))

if not recall_path:
    print("警告: detection_recall_path 为空")
    sys.exit(1)

# 获取迭代信息
log_interval = record.get('log_interval', 100)
total_iterations = record.get('total_iterations', len(recall_path) * log_interval)

# 生成 x 轴（iterations）
x_axis = [i * log_interval for i in range(len(recall_path))]

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 绘制 Recall 曲线
ax1.plot(x_axis, recall_path, 'b-o', linewidth=2, markersize=5, label='Detection Recall')
ax1.set_xlabel('Iterations', fontsize=FONTSIZE)
ax1.set_ylabel('Recall', fontsize=FONTSIZE)
ax1.set_title('Attack Detection Recall', fontsize=FONTSIZE + 2)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=FONTSIZE - 2)
ax1.tick_params(labelsize=FONTSIZE - 2)
ax1.set_ylim([0, 1.05])

# 绘制 Precision 曲线（如果有）
if precision_path and len(precision_path) > 0:
    ax2.plot(x_axis, precision_path, 'r-s', linewidth=2, markersize=5, label='Detection Precision')
    ax2.set_xlabel('Iterations', fontsize=FONTSIZE)
    ax2.set_ylabel('Precision', fontsize=FONTSIZE)
    ax2.set_title('Attack Detection Precision', fontsize=FONTSIZE + 2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=FONTSIZE - 2)
    ax2.tick_params(labelsize=FONTSIZE - 2)
    ax2.set_ylim([0, 1.05])
else:
    ax2.text(0.5, 0.5, 'No Precision Data', 
            horizontalalignment='center', verticalalignment='center',
            transform=ax2.transAxes, fontsize=FONTSIZE)
    ax2.set_title('Attack Detection Precision', fontsize=FONTSIZE + 2)

# 调整布局
plt.tight_layout()

# 保存图片
pic_dir = os.path.join(__FILE_DIR__, 'pic')
if not os.path.isdir(pic_dir):
    os.makedirs(pic_dir)

pic_name = f'{title}_detection_recall'
pic_path = os.path.join(pic_dir, pic_name + '.png')
plt.savefig(pic_path, dpi=300, bbox_inches='tight')
print(f"检测 recall 曲线已保存到: {pic_path}")

# 同时保存 PDF 格式
pdf_path = os.path.join(pic_dir, pic_name + '.pdf')
plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
print(f"PDF 已保存到: {pdf_path}")

# 显示图片
plt.show()
