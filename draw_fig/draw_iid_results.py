import matplotlib.pyplot as plt
import os
import sys

# 添加项目根目录到路径
__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(__FILE_DIR__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path, isfile_in_cache

# 设置缓存路径
__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(project_root, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)

# 配置参数
method = 'CMomentum'
task_name = 'NeuralNetwork'
dataset = 'cifar10'
partition_name = 'iidPartition'  # IID 环境
graph_name = 'Centralized_n=10_b=3'  # b=3 表示3个恶意节点

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

def try_load_file(file_name, path_list):
    """尝试加载文件，如果失败返回None"""
    try:
        if isfile_in_cache(file_name, path_list):
            return load_file_in_cache(file_name, path_list)
    except Exception as e:
        print(f"无法加载文件 {file_name}: {e}")
    return None

def draw_iid_baseline():
    """绘制 IID 环境下无攻击（baseline）的训练曲线"""
    taskname = f'{task_name}_{dataset}'
    file_path = [taskname, graph_name, partition_name]
    
    # 定义要对比的聚合方法（baseline 场景）
    aggregations_to_load = [
        {
            'name': 'Mean',
            'file_name': f'{method}_baseline_mean',
            'color': 'red',
            'marker': 'o',
            'linestyle': '-',
            'linewidth': 2
        },
        {
            'name': 'HSM-Adaptive',
            'file_name': f'{method}_baseline_HSM-Adaptive',
            'color': 'blue',
            'marker': 's',
            'linestyle': '-',
            'linewidth': 2
        },
        {
            'name': 'HSM-FedAvg',
            'file_name': f'{method}_baseline_HSM-FedAvg',
            'color': 'green',
            'marker': '^',
            'linestyle': '-',
            'linewidth': 2
        }
    ]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 加载并绘制每个聚合方法的结果
    for agg_config in aggregations_to_load:
        record = try_load_file(agg_config['file_name'], file_path)
        
        if record is None:
            print(f"警告: 无法加载 {agg_config['file_name']}")
            continue
        
        # 提取数据
        acc_path = record.get('acc_path', None)
        if acc_path is None:
            acc_path = record
        
        # 转换数据格式
        acc_path = convert_to_list(acc_path)
        
        # 获取迭代信息
        display_interval = record.get('display_interval', 100)
        rounds = record.get('rounds', len(acc_path) - 1)
        
        # 生成x轴（iterations）
        x_axis = [r * display_interval for r in range(rounds + 1)]
        
        # 确保 x_axis 和 acc_path 长度一致
        min_len = min(len(x_axis), len(acc_path))
        x_axis = x_axis[:min_len]
        acc_path = acc_path[:min_len]
        
        # 绘制曲线
        ax.plot(x_axis, acc_path, 
               color=agg_config['color'],
               marker=agg_config['marker'],
               linestyle=agg_config['linestyle'],
               linewidth=agg_config['linewidth'],
               markersize=6,
               markevery=max(1, len(x_axis) // 20),
               label=agg_config['name'])
        
        print(f"成功加载并绘制: {agg_config['name']} ({len(acc_path)} 个数据点)")
    
    # 设置标签和标题
    ax.set_xlabel('Iterations', fontsize=FONTSIZE)
    ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
    title = f'CIFAR-10 - IID Partition - Baseline (No Attack)'
    ax.set_title(title, fontsize=FONTSIZE + 2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTSIZE - 2, loc='best')
    ax.tick_params(labelsize=FONTSIZE - 2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    pic_dir = os.path.join(__FILE_DIR__, 'pic')
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    
    pic_name = f'{method}_{dataset}_iid_baseline_b3'
    pic_path = os.path.join(pic_dir, pic_name + '.png')
    plt.savefig(pic_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存到: {pic_path}")
    
    # 同时保存 PDF 格式
    pdf_path = os.path.join(pic_dir, pic_name + '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"PDF 已保存到: {pdf_path}")
    
    # 显示图片
    plt.show()

def draw_iid_label_flipping():
    """绘制 IID 环境下 label flipping 攻击的训练曲线"""
    taskname = f'{task_name}_{dataset}'
    file_path = [taskname, graph_name, partition_name]
    
    # 定义要对比的聚合方法
    aggregations_to_load = [
        {
            'name': 'Mean',
            'file_name': f'{method}_label_flipping_mean',
            'color': 'red',
            'marker': 'o',
            'linestyle': '-',
            'linewidth': 2
        },
        {
            'name': 'HSM-FedAvg',
            'file_name': f'{method}_label_flipping_HSM-FedAvg',
            'color': 'blue',
            'marker': 's',
            'linestyle': '-',
            'linewidth': 2
        },
        {
            'name': 'HSM-Adaptive',
            'file_name': f'{method}_label_flipping_HSM-Adaptive',
            'color': 'green',
            'marker': '^',
            'linestyle': '-',
            'linewidth': 2
        }
    ]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 加载并绘制每个聚合方法的结果
    for agg_config in aggregations_to_load:
        record = try_load_file(agg_config['file_name'], file_path)
        
        if record is None:
            print(f"警告: 无法加载 {agg_config['file_name']}")
            continue
        
        # 提取数据
        acc_path = record.get('acc_path', None)
        if acc_path is None:
            acc_path = record
        
        # 转换数据格式
        acc_path = convert_to_list(acc_path)
        
        # 获取迭代信息
        display_interval = record.get('display_interval', 100)
        rounds = record.get('rounds', len(acc_path) - 1)
        
        # 生成x轴（iterations）
        x_axis = [r * display_interval for r in range(rounds + 1)]
        
        # 确保 x_axis 和 acc_path 长度一致
        min_len = min(len(x_axis), len(acc_path))
        x_axis = x_axis[:min_len]
        acc_path = acc_path[:min_len]
        
        # 绘制曲线
        ax.plot(x_axis, acc_path, 
               color=agg_config['color'],
               marker=agg_config['marker'],
               linestyle=agg_config['linestyle'],
               linewidth=agg_config['linewidth'],
               markersize=6,
               markevery=max(1, len(x_axis) // 20),
               label=agg_config['name'])
        
        print(f"成功加载并绘制: {agg_config['name']} ({len(acc_path)} 个数据点)")
    
    # 设置标签和标题
    ax.set_xlabel('Iterations', fontsize=FONTSIZE)
    ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
    title = f'CIFAR-10 - IID Partition - Label Flipping Attack (b=3)'
    ax.set_title(title, fontsize=FONTSIZE + 2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTSIZE - 2, loc='best')
    ax.tick_params(labelsize=FONTSIZE - 2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    pic_dir = os.path.join(__FILE_DIR__, 'pic')
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    
    pic_name = f'{method}_{dataset}_label_flipping_iid_b3'
    pic_path = os.path.join(pic_dir, pic_name + '.png')
    plt.savefig(pic_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存到: {pic_path}")
    
    # 同时保存 PDF 格式
    pdf_path = os.path.join(pic_dir, pic_name + '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"PDF 已保存到: {pdf_path}")
    
    # 显示图片
    plt.show()

if __name__ == '__main__':
    # 选择绘制场景
    # 'baseline' - 无攻击场景
    # 'label_flipping' - label flipping 攻击场景
    mode = 'label_flipping'  # 或 'baseline'
    
    if mode == 'baseline':
        draw_iid_baseline()
    elif mode == 'label_flipping':
        draw_iid_label_flipping()
    else:
        print(f"未知模式: {mode}，请选择 'baseline' 或 'label_flipping'")
