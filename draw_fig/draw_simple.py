import matplotlib.pyplot as plt
import os
import sys

# 添加项目根目录到路径（支持从项目根目录或 draw_fig 目录运行）
__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(__FILE_DIR__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path, isfile_in_cache


__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(project_root, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)


method = 'CMomentum'
attack_name = 'MSA'  
aggregation_name = 'mean'
dataset = 'cifar10' 
task_name = 'NeuralNetwork'
partition_name = 'LabelSeperation' 
graph_name = 'Centralized_n=10_b=1'

# 绘图样式
FONTSIZE = 20

def try_load_file(file_name, path_list):
    """尝试加载文件，如果失败返回None"""
    try:
        if isfile_in_cache(file_name, path_list):
            return load_file_in_cache(file_name, path_list)
    except Exception as e:
        print(f"无法加载文件 {file_name}: {e}")
    return None

def draw_single_result():
    """绘制单个结果"""
    taskname = f'{task_name}_{dataset}'
    file_path = [taskname, graph_name, partition_name]
    

    possible_names = [
        f'{method}_{attack_name}_{attack_name}_{aggregation_name}',  # CMomentum_MSA_MSA_mean (实际格式)
        f'{method}_{attack_name}_{aggregation_name}',  # CMomentum_MSA_mean (标准格式)
        f'{method}_{attack_name}_mean',  # 备用格式
    ]
    
    record = None
    used_filename = None
    
    for filename in possible_names:
        record = try_load_file(filename, file_path)
        if record is not None:
            used_filename = filename
            print(f"成功加载文件: {filename}")
            break
    
    if record is None:
        print(f"错误: 无法找到数据文件")
        print(f"尝试的路径: {os.path.join(__CACHE_PATH__, *file_path)}")
        print(f"尝试的文件名: {possible_names}")
        return
    
    # 提取数据
    acc_path = record['acc_path']
    display_interval = record['display_interval']
    rounds = record['rounds']
    
    # 处理 CUDA tensor 或 numpy 数组，转换为 Python 列表
    import torch
    import numpy as np
    
    # 如果是 tensor，先转换为 numpy
    if isinstance(acc_path, torch.Tensor):
        if acc_path.is_cuda:
            acc_path = acc_path.detach().cpu().numpy()
        else:
            acc_path = acc_path.detach().numpy()
    
    # 如果是 numpy 数组，转换为列表
    if isinstance(acc_path, np.ndarray):
        acc_path = acc_path.tolist()
    
    # 如果是列表但包含 tensor，转换每个元素
    if isinstance(acc_path, list):
        acc_path = [float(item.cpu().item() if isinstance(item, torch.Tensor) and item.is_cuda 
                         else item.item() if isinstance(item, torch.Tensor) 
                         else float(item)) for item in acc_path]
    
    # 生成x轴（iterations）
    x_axis = [r * display_interval for r in range(rounds + 1)]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制准确率曲线
    ax.plot(x_axis, acc_path, '-o', linewidth=2, markersize=5, label=f'{attack_name} + {aggregation_name}')
    
    # 设置标签和标题
    ax.set_xlabel('Iterations', fontsize=FONTSIZE)
    ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
    title = f'{dataset.upper()} - {partition_name} - {attack_name}'
    ax.set_title(title, fontsize=FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE-2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    pic_dir = os.path.join(__FILE_DIR__, 'pic')
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    
    pic_name = f'{method}_{dataset}_{attack_name}_{aggregation_name}_{partition_name}'
    pic_path = os.path.join(pic_dir, pic_name + '.png')
    plt.savefig(pic_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {pic_path}")
    
    # 显示图片
    plt.show()

def draw_comparison():
    """绘制对比图（如果有多个结果）"""
    datasets_to_try = ['cifar10', 'mnist']
    attacks_to_try = ['MSA', 'MSA_MSA']
    aggregations_to_try = ['mean']
    
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    ax = axes
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'v', 'x']
    color_idx = 0
    
    for dataset_name in datasets_to_try:
        taskname = f'{task_name}_{dataset_name}'
        file_path = [taskname, graph_name, partition_name]
        
        for agg_name in aggregations_to_try:
            for attack in attacks_to_try:
                # 尝试加载文件
                filename = f'{method}_{attack}_{agg_name}'
                record = try_load_file(filename, file_path)
                
                if record is None:
                    # 尝试 MSA_MSA 格式
                    filename = f'{method}_{attack}_{attack}_{agg_name}'
                    record = try_load_file(filename, file_path)
                
                if record is not None:
                    acc_path = record['acc_path']
                    display_interval = record['display_interval']
                    rounds = record['rounds']
                    
                    # 处理 CUDA tensor 或 numpy 数组
                    import torch
                    if isinstance(acc_path, torch.Tensor):
                        if acc_path.is_cuda:
                            acc_path = acc_path.cpu().numpy()
                        else:
                            acc_path = acc_path.numpy()
                    elif hasattr(acc_path, 'tolist'):
                        acc_path = acc_path.tolist()
                    
                    x_axis = [r * display_interval for r in range(rounds + 1)]
                    
                    label = f'{dataset_name.upper()} - {attack}'
                    ax.plot(x_axis, acc_path, '-', 
                           color=colors[color_idx % len(colors)],
                           marker=markers[color_idx % len(markers)],
                           linewidth=2, markersize=5, label=label)
                    color_idx += 1
    
    ax.set_xlabel('Iterations', fontsize=FONTSIZE)
    ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
    ax.set_title(f'{partition_name} - Comparison', fontsize=FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTSIZE-2)
    ax.tick_params(labelsize=FONTSIZE-2)
    
    plt.tight_layout()
    
    # 保存
    pic_dir = os.path.join(__FILE_DIR__, 'pic')
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    pic_path = os.path.join(pic_dir, 'comparison.png')
    plt.savefig(pic_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存到: {pic_path}")
    plt.show()

if __name__ == '__main__':
    # 选择绘制方式
    mode = 'single'  # 或者 'comparison'
    
    if mode == 'single':
        draw_single_result()
    else:
        draw_comparison()
