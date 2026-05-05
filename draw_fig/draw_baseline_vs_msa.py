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
aggregation_name = 'mean'
task_name = 'NeuralNetwork'
partition_name = 'LabelSeperation'  # 对应 noniid
graph_name_baseline = 'Centralized_n=10_b=0'
graph_name_attack = 'Centralized_n=10_b=1'

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

def draw_baseline_vs_msa(dataset='cifar10'):
    """绘制 Baseline vs MSA 攻击对比图"""
    taskname = f'{task_name}_{dataset}'
    
    # 加载 Baseline（无攻击）- Mean
    baseline_file_name = f'{method}_baseline_{aggregation_name}'
    baseline_path = [taskname, graph_name_baseline, partition_name]
    baseline_record = try_load_file(baseline_file_name, baseline_path)
    
    # 加载 Baseline（无攻击）- HSM-FedAvg
    baseline_hsm_file_name = f'{method}_baseline_HSM-FedAvg'
    baseline_hsm_record = try_load_file(baseline_hsm_file_name, baseline_path)
    
    # 加载 MSA 攻击结果（尝试多种文件名格式）
    msa_file_name = None
    msa_record = None
    msa_path = [taskname, graph_name_attack, partition_name]
    
    # 尝试 MSA_MSA 格式（实际文件名）
    msa_file_name = f'{method}_MSA_MSA_{aggregation_name}'
    msa_record = try_load_file(msa_file_name, msa_path)
    
    if msa_record is None:
        # 尝试标准格式
        msa_file_name = f'{method}_MSA_{aggregation_name}'
        msa_record = try_load_file(msa_file_name, msa_path)
    
    # 检查数据是否加载成功
    if baseline_record is None:
        print(f"警告: 无法加载 Baseline 数据")
        print(f"  路径: {os.path.join(__CACHE_PATH__, *baseline_path)}")
        print(f"  文件名: {baseline_file_name}")
        if msa_record is None:
            print(f"错误: 也无法加载 MSA 攻击数据")
            return
        print(f"  只绘制 MSA 攻击结果")
    
    if msa_record is None:
        print(f"警告: 无法加载 MSA 攻击数据")
        print(f"  路径: {os.path.join(__CACHE_PATH__, *msa_path)}")
        if baseline_record is None:
            print(f"错误: 也无法加载 Baseline 数据")
            return
        print(f"  只绘制 Baseline 结果")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制 Baseline (Mean)
    if baseline_record is not None:
        baseline_acc = convert_to_list(baseline_record['acc_path'])
        baseline_display_interval = baseline_record['display_interval']
        baseline_rounds = baseline_record['rounds']
        baseline_x = [r * baseline_display_interval for r in range(baseline_rounds + 1)]
        
        ax.plot(baseline_x, baseline_acc, '-o', 
               color='blue', linewidth=2, markersize=5, 
               label='Baseline (Mean)', alpha=0.8)
        print(f"[OK] 已加载 Baseline (Mean): {len(baseline_acc)} 个数据点")
    
    # 绘制 Baseline (HSM-FedAvg)
    if baseline_hsm_record is not None:
        baseline_hsm_acc = convert_to_list(baseline_hsm_record['acc_path'])
        baseline_hsm_display_interval = baseline_hsm_record['display_interval']
        baseline_hsm_rounds = baseline_hsm_record['rounds']
        baseline_hsm_x = [r * baseline_hsm_display_interval for r in range(baseline_hsm_rounds + 1)]
        
        ax.plot(baseline_hsm_x, baseline_hsm_acc, '--o', 
               color='navy', linewidth=2, markersize=5, 
               label='Baseline (HSM-FedAvg)', alpha=0.8)
        print(f"[OK] 已加载 Baseline (HSM-FedAvg): {len(baseline_hsm_acc)} 个数据点")
    
    # 绘制 MSA 攻击
    if msa_record is not None:
        msa_acc = convert_to_list(msa_record['acc_path'])
        msa_display_interval = msa_record['display_interval']
        msa_rounds = msa_record['rounds']
        msa_x = [r * msa_display_interval for r in range(msa_rounds + 1)]
        
        ax.plot(msa_x, msa_acc, '-s', 
               color='red', linewidth=2, markersize=5, 
               label='MSA Attack', alpha=0.8)
        print(f"[OK] 已加载 MSA Attack: {len(msa_acc)} 个数据点")
    
    # 设置标签和标题
    ax.set_xlabel('Iterations', fontsize=FONTSIZE)
    ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
    title = f'{dataset.upper()} - {partition_name} - Baseline vs MSA Attack'
    ax.set_title(title, fontsize=FONTSIZE)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=FONTSIZE-2, loc='best')
    ax.tick_params(labelsize=FONTSIZE-2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    pic_dir = os.path.join(__FILE_DIR__, 'pic')
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    
    pic_name = f'{method}_{dataset}_baseline_vs_MSA_{partition_name}'
    pic_path = os.path.join(pic_dir, pic_name + '.png')
    plt.savefig(pic_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存到: {pic_path}")
    
    # 显示图片
    plt.show()

def draw_multi_dataset_comparison():
    """绘制多个数据集的对比图"""
    datasets = ['cifar10', 'mnist']
    fig, axes = plt.subplots(1, len(datasets), figsize=(16, 6))
    
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        taskname = f'{task_name}_{dataset}'
        
        # 加载 Baseline (Mean)
        baseline_file_name = f'{method}_baseline_{aggregation_name}'
        baseline_path = [taskname, graph_name_baseline, partition_name]
        baseline_record = try_load_file(baseline_file_name, baseline_path)
        
        # 加载 Baseline (HSM-FedAvg)
        baseline_hsm_file_name = f'{method}_baseline_HSM-FedAvg'
        baseline_hsm_record = try_load_file(baseline_hsm_file_name, baseline_path)
        
        # 加载 MSA 攻击
        msa_file_name = f'{method}_MSA_MSA_{aggregation_name}'
        msa_path = [taskname, graph_name_attack, partition_name]
        msa_record = try_load_file(msa_file_name, msa_path)
        
        if msa_record is None:
            msa_file_name = f'{method}_MSA_{aggregation_name}'
            msa_record = try_load_file(msa_file_name, msa_path)
        
        # 绘制 Baseline (Mean)
        if baseline_record is not None:
            baseline_acc = convert_to_list(baseline_record['acc_path'])
            baseline_display_interval = baseline_record['display_interval']
            baseline_rounds = baseline_record['rounds']
            baseline_x = [r * baseline_display_interval for r in range(baseline_rounds + 1)]
            ax.plot(baseline_x, baseline_acc, '-o', 
                   color='blue', linewidth=2, markersize=4, 
                   label='Baseline (Mean)', alpha=0.8)
        
        # 绘制 Baseline (HSM-FedAvg)
        if baseline_hsm_record is not None:
            baseline_hsm_acc = convert_to_list(baseline_hsm_record['acc_path'])
            baseline_hsm_display_interval = baseline_hsm_record['display_interval']
            baseline_hsm_rounds = baseline_hsm_record['rounds']
            baseline_hsm_x = [r * baseline_hsm_display_interval for r in range(baseline_hsm_rounds + 1)]
            ax.plot(baseline_hsm_x, baseline_hsm_acc, '--o', 
                   color='navy', linewidth=2, markersize=4, 
                   label='Baseline (HSM-FedAvg)', alpha=0.8)
        
        # 绘制 MSA 攻击
        if msa_record is not None:
            msa_acc = convert_to_list(msa_record['acc_path'])
            msa_display_interval = msa_record['display_interval']
            msa_rounds = msa_record['rounds']
            msa_x = [r * msa_display_interval for r in range(msa_rounds + 1)]
            ax.plot(msa_x, msa_acc, '-s', 
                   color='red', linewidth=2, markersize=4, 
                   label='MSA Attack', alpha=0.8)
        
        ax.set_xlabel('Iterations', fontsize=FONTSIZE)
        if idx == 0:
            ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
        ax.set_title(f'{dataset.upper()}', fontsize=FONTSIZE)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=FONTSIZE-3, loc='best')
        ax.tick_params(labelsize=FONTSIZE-3)
    
    plt.suptitle(f'Baseline vs MSA Attack - {partition_name}', fontsize=FONTSIZE+2)
    plt.tight_layout()
    
    # 保存
    pic_dir = os.path.join(__FILE_DIR__, 'pic')
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    pic_path = os.path.join(pic_dir, f'{method}_baseline_vs_MSA_multidataset_{partition_name}.png')
    plt.savefig(pic_path, dpi=300, bbox_inches='tight')
    print(f"多数据集对比图已保存到: {pic_path}")
    plt.show()

if __name__ == '__main__':
    # 选择绘制方式
    mode = 'single'  # 'single' 或 'multi'
    
    if mode == 'single':
        # 绘制单个数据集的对比（默认 cifar10）
        dataset = 'cifar10'  # 或改为 'mnist'
        draw_baseline_vs_msa(dataset=dataset)
    else:
        # 绘制多个数据集的对比
        draw_multi_dataset_comparison()
