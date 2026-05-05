"""
自动绘制 IID 环境下所有攻击场景的 HSM vs Mean 对比图
从指定目录加载所有可用的数据文件并绘制
"""
import matplotlib.pyplot as plt
import os
import sys
import glob

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

def scan_available_files(path_list):
    """
    扫描指定路径下所有可用的记录文件
    
    Returns:
        dict: {attack_type: [file_names]}
    """
    taskname = f'{task_name}_{dataset}'
    file_path = [taskname, graph_name, partition_name]
    
    # 构建完整路径
    full_path = os.path.join(__CACHE_PATH__, *file_path)
    
    # 扫描所有文件（排除日志文件）
    available_files = {}
    if os.path.exists(full_path):
        for file_name in os.listdir(full_path):
            if file_name.endswith('.txt'):
                continue  # 跳过日志文件
            
            # 解析文件名：CMomentum_{attack}_{aggregation}
            parts = file_name.split('_')
            if len(parts) >= 3:
                # 提取攻击类型和聚合方法
                if parts[1] == 'label' and parts[2] == 'flipping':
                    attack_type = 'label_flipping'
                    aggregation = '_'.join(parts[3:]) if len(parts) > 3 else 'unknown'
                elif parts[1] == 'HisMSA':
                    attack_type = 'HisMSA'
                    # HisMSA_HisMSA_HSM-Adaptive -> HSM-Adaptive
                    if len(parts) >= 4 and parts[2] == 'HisMSA':
                        aggregation = '_'.join(parts[3:]) if len(parts) > 3 else 'unknown'
                    else:
                        aggregation = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
                elif parts[1] == 'MSA':
                    attack_type = 'MSA'
                    # MSA_MSA_HSM-Adaptive -> HSM-Adaptive
                    if len(parts) >= 4 and parts[2] == 'MSA':
                        aggregation = '_'.join(parts[3:]) if len(parts) > 3 else 'unknown'
                    else:
                        aggregation = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
                elif parts[1] == 'baseline':
                    attack_type = 'baseline'
                    aggregation = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
                else:
                    attack_type = 'unknown'
                    aggregation = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown'
                
                if attack_type not in available_files:
                    available_files[attack_type] = []
                available_files[attack_type].append({
                    'file_name': file_name,
                    'aggregation': aggregation
                })
    
    return available_files, file_path

def draw_attack_comparison(attack_type, files_info, path_list):
    """
    绘制特定攻击场景的对比图
    
    Args:
        attack_type: 攻击类型（如 'label_flipping', 'HisMSA', 'MSA'）
        files_info: 该攻击类型下的文件信息列表
        path_list: 缓存路径列表
    """
    # 定义聚合方法的显示配置
    aggregation_configs = {
        'mean': {
            'name': 'Mean',
            'color': 'red',
            'marker': 'o',
            'linestyle': '-',
            'linewidth': 2
        },
        'HSM-Adaptive': {
            'name': 'HSM-Adaptive',
            'color': 'blue',
            'marker': 's',
            'linestyle': '-',
            'linewidth': 2
        },
        'HSM-FedAvg': {
            'name': 'HSM-FedAvg',
            'color': 'green',
            'marker': '^',
            'linestyle': '-',
            'linewidth': 2
        }
    }
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 加载并绘制每个聚合方法的结果
    plotted_count = 0
    for file_info in files_info:
        file_name = file_info['file_name']
        aggregation = file_info['aggregation']
        
        # 确定显示配置
        if aggregation in aggregation_configs:
            config = aggregation_configs[aggregation]
        else:
            # 默认配置
            config = {
                'name': aggregation,
                'color': 'gray',
                'marker': 'x',
                'linestyle': '--',
                'linewidth': 2
            }
        
        record = try_load_file(file_name, path_list)
        
        if record is None:
            print(f"警告: 无法加载 {file_name}")
            continue
        
        # 提取数据
        acc_path = record.get('acc_path', None)
        if acc_path is None:
            acc_path = record
        
        # 转换数据格式
        acc_path = convert_to_list(acc_path)
        
        if not acc_path or len(acc_path) == 0:
            print(f"警告: {file_name} 没有有效数据")
            continue
        
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
               color=config['color'],
               marker=config['marker'],
               linestyle=config['linestyle'],
               linewidth=config['linewidth'],
               markersize=6,
               markevery=max(1, len(x_axis) // 20),
               label=config['name'])
        
        print(f"成功加载并绘制: {config['name']} ({len(acc_path)} 个数据点)")
        plotted_count += 1
    
    if plotted_count == 0:
        print(f"警告: {attack_type} 场景下没有可用的数据")
        plt.close(fig)
        return
    
    # 设置标签和标题
    ax.set_xlabel('Iterations', fontsize=FONTSIZE)
    ax.set_ylabel('Accuracy', fontsize=FONTSIZE)
    
    # 生成标题
    attack_display_names = {
        'label_flipping': 'Label Flipping Attack',
        'HisMSA': 'HisMSA Attack',
        'MSA': 'MSA Attack',
        'baseline': 'Baseline (No Attack)'
    }
    attack_display = attack_display_names.get(attack_type, attack_type)
    title = f'CIFAR-10 - IID Partition - {attack_display} (b=3)'
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
    
    pic_name = f'{method}_{dataset}_{attack_type}_iid_b3_comparison'
    pic_path = os.path.join(pic_dir, pic_name + '.png')
    plt.savefig(pic_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存到: {pic_path}")
    
    # 同时保存 PDF 格式
    pdf_path = os.path.join(pic_dir, pic_name + '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"PDF 已保存到: {pdf_path}")
    
    # 显示图片
    plt.show()

def draw_all_comparisons():
    """扫描并绘制所有可用的攻击场景对比图"""
    # 扫描可用文件
    available_files, path_list = scan_available_files(None)
    
    if not available_files:
        print("错误: 未找到任何可用的记录文件")
        return
    
    print(f"找到 {len(available_files)} 个攻击场景:")
    for attack_type, files in available_files.items():
        print(f"  - {attack_type}: {len(files)} 个文件")
        for f in files:
            print(f"    * {f['file_name']} ({f['aggregation']})")
    
    # 按优先级排序：label_flipping > HisMSA > MSA > baseline > others
    priority_order = ['label_flipping', 'HisMSA', 'MSA', 'baseline']
    sorted_attacks = sorted(available_files.keys(), 
                          key=lambda x: (priority_order.index(x) if x in priority_order else 999, x))
    
    # 绘制每个攻击场景
    for attack_type in sorted_attacks:
        print(f"\n{'='*60}")
        print(f"绘制 {attack_type} 场景对比图...")
        print(f"{'='*60}")
        draw_attack_comparison(attack_type, available_files[attack_type], path_list)

if __name__ == '__main__':
    draw_all_comparisons()
