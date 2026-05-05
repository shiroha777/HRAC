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
graph_name_attack = 'Centralized_n=10_b=3'  # 修改为 b=3，对应3个恶意节点

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

def draw_attack_comparison(dataset='cifar10'):
    """绘制多种攻击的对比图：Baseline vs MSA vs Label Flipping"""
    taskname = f'{task_name}_{dataset}'
    
    # 定义要对比的攻击类型
    attacks_to_load = [
        {
            'name': 'Baseline (Mean)',
            'file_name': f'{method}_baseline_{aggregation_name}',
            'path': [taskname, graph_name_baseline, partition_name],
            'color': 'blue',
            'marker': 'o',
            'linestyle': '-'
        },
        {
            'name': 'Baseline (HSM-FedAvg)',
            'file_name': f'{method}_baseline_HSM-FedAvg',
            'path': [taskname, graph_name_baseline, partition_name],
            'color': 'navy',
            'marker': 'o',
            'linestyle': '--'
        },
        {
            'name': 'MSA Attack',
            'file_name': f'{method}_MSA_MSA_{aggregation_name}',  # 尝试MSA_MSA格式
            'alt_file_name': f'{method}_MSA_{aggregation_name}',  # 备用标准格式
            'path': [taskname, graph_name_attack, partition_name],
            'color': 'red',
            'marker': 's',
            'linestyle': '-'
        },
        {
            'name': 'Label Flipping Attack',
            'file_name': f'{method}_label_flipping_{aggregation_name}',
            'path': [taskname, graph_name_attack, partition_name],  # 使用 b=3 的路径
            'color': 'green',
            'marker': '^',
            'linestyle': '-'
        },
        {
            'name': 'MSA Attack + CC Defense',
            'file_name': f'{method}_MSA_MSA_CC_tau=0.3',  # CC with threshold=0.3 for noniid
            'path': [taskname, graph_name_attack, partition_name],
            'color': 'purple',
            'marker': 'd',
            'linestyle': '-'
        },
        {
            'name': 'HisMSA Attack',
            'file_name': f'{method}_HisMSA_HisMSA_{aggregation_name}',
            'path': [taskname, graph_name_attack, partition_name],
            'color': 'orange',
            'marker': 'v',
            'linestyle': '-'
        },
        {
            'name': 'HisMSA Attack + CC Defense',
            'file_name': f'{method}_HisMSA_HisMSA_CC_tau=0.3',  # CC with threshold=0.3 for noniid
            'path': [taskname, graph_name_attack, partition_name],
            'color': 'brown',
            'marker': 'p',
            'linestyle': '-'
        },
        {
            'name': 'HisMSA Attack + FABA Defense',
            'file_name': f'{method}_HisMSA_HisMSA_faba',
            'path': [taskname, graph_name_attack, partition_name],
            'color': 'magenta',
            'marker': 'x',
            'linestyle': '-'
        },
        {
            'name': 'Label Flipping + HSM-FedAvg Defense',
            'file_name': f'{method}_label_flipping_HSM-FedAvg',
            'path': [taskname, graph_name_attack, partition_name],
            'color': 'cyan',
            'marker': '*',
            'linestyle': '-'
        },
        {
            'name': 'MSA Attack + HSM-FedAvg Defense',
            'file_name': f'{method}_MSA_MSA_HSM-FedAvg',
            'path': [taskname, graph_name_attack, partition_name],
            'color': 'teal',
            'marker': 'h',
            'linestyle': '--'
        },
        {
            'name': 'HisMSA Attack + HSM-FedAvg Defense',
            'file_name': f'{method}_HisMSA_HisMSA_HSM-FedAvg',
            'path': [taskname, graph_name_attack, partition_name],
            'color': 'coral',
            'marker': 'D',
            'linestyle': '--'
        }
    ]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 7))
    
    loaded_count = 0
    
    # 加载并绘制每个攻击的结果
    for attack_config in attacks_to_load:
        record = None
        
        # 尝试加载主要文件名
        record = try_load_file(attack_config['file_name'], attack_config['path'])
        
        # 如果失败且有关备用文件名，尝试备用文件名（仅用于MSA）
        if record is None and 'alt_file_name' in attack_config:
            record = try_load_file(attack_config['alt_file_name'], attack_config['path'])
            if record is not None:
                print(f"[OK] 使用备用文件名加载: {attack_config['alt_file_name']}")
        
        if record is not None:
            acc_path = convert_to_list(record['acc_path'])
            display_interval = record['display_interval']
            rounds = record['rounds']
            x_axis = [r * display_interval for r in range(rounds + 1)]
            
            ax.plot(x_axis, acc_path, 
                   color=attack_config['color'],
                   marker=attack_config['marker'],
                   linestyle=attack_config['linestyle'],
                   linewidth=2, 
                   markersize=5,
                   label=attack_config['name'],
                   alpha=0.8,
                   markevery=max(1, len(x_axis)//20))  # 每20个点标记一次
            
            print(f"[OK] 已加载 {attack_config['name']}: {len(acc_path)} 个数据点")
            loaded_count += 1
        else:
            print(f"[WARN] 无法加载 {attack_config['name']}")
            print(f"  尝试的路径: {os.path.join(__CACHE_PATH__, *attack_config['path'])}")
            print(f"  尝试的文件名: {attack_config['file_name']}")
    
    if loaded_count == 0:
        print(f"[ERROR] 无法加载任何数据文件！")
        return
    
    # 设置标签和标题
    ax.set_xlabel('Iterations', fontsize=FONTSIZE)
    ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
    title = f'{dataset.upper()} - {partition_name} - Attack Comparison'
    ax.set_title(title, fontsize=FONTSIZE+2, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=FONTSIZE-1, loc='best', framealpha=0.9)
    ax.tick_params(labelsize=FONTSIZE-2)
    
    # 设置y轴范围，确保能看到所有曲线
    ax.set_ylim([0, 1])  # 可以根据实际情况调整
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    pic_dir = os.path.join(__FILE_DIR__, 'pic')
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    
    pic_name = f'{method}_{dataset}_attack_comparison_{partition_name}'
    pic_path_png = os.path.join(pic_dir, pic_name + '.png')
    pic_path_pdf = os.path.join(pic_dir, pic_name + '.pdf')
    
    plt.savefig(pic_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(pic_path_pdf, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 图片已保存到:")
    print(f"  PNG: {pic_path_png}")
    print(f"  PDF: {pic_path_pdf}")
    
    # 显示图片
    plt.show()

def draw_multi_dataset_attack_comparison():
    """绘制多个数据集的攻击对比图"""
    datasets = ['cifar10', 'mnist']
    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6))
    
    if len(datasets) == 1:
        axes = [axes]
    
    attacks_config = [
        {'name': 'Baseline', 'file': f'{method}_baseline_mean', 'path_type': 'baseline',
         'color': 'blue', 'marker': 'o'},
        {'name': 'MSA', 'file': f'{method}_MSA_MSA_mean', 'alt_file': f'{method}_MSA_mean',
         'path_type': 'attack', 'color': 'red', 'marker': 's'},
        {'name': 'Label Flipping', 'file': f'{method}_label_flipping_mean',
         'path_type': 'attack', 'color': 'green', 'marker': '^'},
        {'name': 'MSA + CC', 'file': f'{method}_MSA_MSA_CC_tau=0.3',
         'path_type': 'attack', 'color': 'purple', 'marker': 'd'},
        {'name': 'HisMSA', 'file': f'{method}_HisMSA_HisMSA_mean',
         'path_type': 'attack', 'color': 'orange', 'marker': 'v'},
        {'name': 'HisMSA + CC', 'file': f'{method}_HisMSA_HisMSA_CC_tau=0.3',
         'path_type': 'attack', 'color': 'brown', 'marker': 'p'},
        {'name': 'HisMSA + FABA', 'file': f'{method}_HisMSA_HisMSA_faba',
         'path_type': 'attack', 'color': 'magenta', 'marker': 'x'},
        {'name': 'Label Flipping + HSM-FedAvg', 'file': f'{method}_label_flipping_HSM-FedAvg',
         'path_type': 'attack', 'color': 'cyan', 'marker': '*'},
        {'name': 'MSA + HSM-FedAvg', 'file': f'{method}_MSA_MSA_HSM-FedAvg',
         'path_type': 'attack', 'color': 'teal', 'marker': 'h'},
        {'name': 'HisMSA + HSM-FedAvg', 'file': f'{method}_HisMSA_HisMSA_HSM-FedAvg',
         'path_type': 'attack', 'color': 'coral', 'marker': 'D'}
    ]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        taskname = f'{task_name}_{dataset}'
        
        for attack_config in attacks_config:
            if attack_config['path_type'] == 'baseline':
                path = [taskname, graph_name_baseline, partition_name]
            else:
                path = [taskname, graph_name_attack, partition_name]
            
            record = try_load_file(attack_config['file'], path)
            if record is None and 'alt_file' in attack_config:
                record = try_load_file(attack_config['alt_file'], path)
            
            if record is not None:
                acc_path = convert_to_list(record['acc_path'])
                display_interval = record['display_interval']
                rounds = record['rounds']
                x_axis = [r * display_interval for r in range(rounds + 1)]
                
                ax.plot(x_axis, acc_path, 
                       color=attack_config['color'],
                       marker=attack_config['marker'],
                       linewidth=2, markersize=4,
                       label=attack_config['name'],
                       alpha=0.8,
                       markevery=max(1, len(x_axis)//20))
        
        ax.set_xlabel('Iterations', fontsize=FONTSIZE)
        if idx == 0:
            ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
        ax.set_title(f'{dataset.upper()}', fontsize=FONTSIZE, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=FONTSIZE-2, loc='best', framealpha=0.9)
        ax.tick_params(labelsize=FONTSIZE-3)
        ax.set_ylim([0, 1])
    
    plt.suptitle(f'Attack Comparison - {partition_name}', fontsize=FONTSIZE+3, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    pic_dir = os.path.join(__FILE_DIR__, 'pic')
    if not os.path.isdir(pic_dir):
        os.makedirs(pic_dir)
    
    pic_name = f'{method}_attack_comparison_multidataset_{partition_name}'
    pic_path_png = os.path.join(pic_dir, pic_name + '.png')
    pic_path_pdf = os.path.join(pic_dir, pic_name + '.pdf')
    
    plt.savefig(pic_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(pic_path_pdf, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 多数据集对比图已保存到:")
    print(f"  PNG: {pic_path_png}")
    print(f"  PDF: {pic_path_pdf}")
    plt.show()

if __name__ == '__main__':
    # 选择绘制方式
    mode = 'single'  # 'single' 或 'multi'
    
    if mode == 'single':
        # 绘制单个数据集的攻击对比（默认 cifar10）
        dataset = 'cifar10'  # 或改为 'mnist'
        draw_attack_comparison(dataset=dataset)
    else:
        # 绘制多个数据集的攻击对比
        draw_multi_dataset_attack_comparison()
