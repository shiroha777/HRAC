"""
绘制 b=3 下 mean 和 hsm 在各个攻击下的对比图
每张图包含四条曲线：
1. Mean 在该攻击下的曲线（b=3）
2. HSM 在该攻击下的曲线（b=3）
3. Mean 的 baseline 曲线（b=0）
4. HSM 的 baseline 曲线（b=0）
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os
import sys

__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(__FILE_DIR__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path, isfile_in_cache

__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(project_root, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)

method = 'CMomentum'
task_name = 'NeuralNetwork'
dataset = 'cifar10'
partition_name = 'iidPartition'
graph_name_b0 = 'Centralized_n=10_b=0'
graph_name_b3 = 'Centralized_n=10_b=3'

FONTSIZE = 16
LINEWIDTH = 2.5
MARKERSIZE = 8

# 定义更区分度的配色方案
COLORS = {
    'mean_attack': '#E74C3C',      # 红色 - Mean攻击
    'hsm_attack': '#3498DB',      # 蓝色 - HSM攻击
    'mean_baseline': '#F39C12',   # 橙色 - Mean基线
    'hsm_baseline': '#9B59B6'     # 紫色 - HSM基线
}

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
            record = load_file_in_cache(file_name, path_list)
            if isinstance(record, dict) and 'acc_path' in record:
                acc_data = convert_to_list(record['acc_path'])
                interval = record.get('display_interval', 100)
                rounds = record.get('rounds', len(acc_data)-1)
                return acc_data, interval, rounds
            elif isinstance(record, dict):
                # 尝试其他可能的键
                for key in ['accuracy', 'acc', 'test_acc']:
                    if key in record:
                        acc_data = convert_to_list(record[key])
                        interval = record.get('display_interval', 100)
                        rounds = record.get('rounds', len(acc_data)-1)
                        return acc_data, interval, rounds
            elif isinstance(record, (list, tuple)):
                acc_data = convert_to_list(record)
                return acc_data, 100, len(acc_data)-1
            else:
                print(f"警告: {file_name} 的数据格式未知: {type(record)}")
    except Exception as e:
        print(f"无法加载文件 {file_name}: {e}")
        import traceback
        traceback.print_exc()
    return None, None, None

def draw_attack_comparison(attack_name, attack_display_name):
    """
    绘制特定攻击场景的对比图
    
    Args:
        attack_name: 攻击类型代码名（如 'label_flipping', 'MSA', 'HisMSA'）
        attack_display_name: 攻击类型显示名
    """
    taskname = f'{task_name}_{dataset}'
    
    # 定义文件名的映射
    attack_file_mapping = {
        'label_flipping': 'label_flipping',
        'MSA': 'MSA_MSA',
        'HisMSA': 'HisMSA_HisMSA'
    }
    
    attack_file_suffix = attack_file_mapping.get(attack_name, attack_name)
    
    # 加载数据
    # 1. Mean 在攻击下的结果（b=3）
    mean_attack_file = f'{method}_{attack_file_suffix}_mean'
    mean_attack_path = [taskname, graph_name_b3, partition_name]
    mean_attack_data, mean_interval, mean_rounds = try_load_file(mean_attack_file, mean_attack_path)
    
    # 2. HSM 在攻击下的结果（b=3）
    hsm_attack_file = f'{method}_{attack_file_suffix}_HSM-Adaptive'
    hsm_attack_path = [taskname, graph_name_b3, partition_name]
    hsm_attack_data, hsm_interval, hsm_rounds = try_load_file(hsm_attack_file, hsm_attack_path)
    
    # 如果 HSM-Adaptive 不存在，尝试 HSM-FedAvg
    if hsm_attack_data is None:
        hsm_attack_file_alt = f'{method}_{attack_file_suffix}_HSM-FedAvg'
        hsm_attack_data, hsm_interval, hsm_rounds = try_load_file(hsm_attack_file_alt, hsm_attack_path)
    
    # 3. Mean baseline（b=0）
    mean_baseline_file = f'{method}_baseline_mean'
    mean_baseline_path = [taskname, graph_name_b0, partition_name]
    mean_baseline_data, mean_bl_interval, mean_bl_rounds = try_load_file(mean_baseline_file, mean_baseline_path)
    
    # 4. HSM baseline（b=0）
    hsm_baseline_file = f'{method}_baseline_HSM-Adaptive'
    hsm_baseline_path = [taskname, graph_name_b0, partition_name]
    hsm_baseline_data, hsm_bl_interval, hsm_bl_rounds = try_load_file(hsm_baseline_file, hsm_baseline_path)
    
    # 如果 HSM-Adaptive baseline 不存在，尝试 HSM-FedAvg
    if hsm_baseline_data is None:
        hsm_baseline_file_alt = f'{method}_baseline_HSM-FedAvg'
        hsm_baseline_data, hsm_bl_interval, hsm_bl_rounds = try_load_file(hsm_baseline_file_alt, hsm_baseline_path)
    
    # 检查数据是否加载成功
    print(f"\n数据加载状态:")
    print(f"  Mean 攻击 (b=3): {mean_attack_file} - {'OK' if mean_attack_data else 'FAIL'}")
    if mean_attack_data:
        print(f"    数据点数: {len(mean_attack_data)}, interval: {mean_interval}")
    print(f"  HSM 攻击 (b=3): {hsm_attack_file} - {'OK' if hsm_attack_data else 'FAIL'}")
    if hsm_attack_data:
        print(f"    数据点数: {len(hsm_attack_data)}, interval: {hsm_interval}")
    print(f"  Mean baseline (b=0): {mean_baseline_file} - {'OK' if mean_baseline_data else 'FAIL'}")
    if mean_baseline_data:
        print(f"    数据点数: {len(mean_baseline_data)}, interval: {mean_bl_interval}")
    print(f"  HSM baseline (b=0): {hsm_baseline_file} - {'OK' if hsm_baseline_data else 'FAIL'}")
    if hsm_baseline_data:
        print(f"    数据点数: {len(hsm_baseline_data)}, interval: {hsm_bl_interval}")
    
    if mean_attack_data is None:
        print(f"警告: 无法加载 Mean 攻击数据: {mean_attack_file}")
    if hsm_attack_data is None:
        print(f"警告: 无法加载 HSM 攻击数据: {hsm_attack_file}")
    if mean_baseline_data is None:
        print(f"警告: 无法加载 Mean baseline 数据: {mean_baseline_file}")
    if hsm_baseline_data is None:
        print(f"警告: 无法加载 HSM baseline 数据: {hsm_baseline_file}")
    
    if all(data is None for data in [mean_attack_data, hsm_attack_data, mean_baseline_data, hsm_baseline_data]):
        print(f"错误: 无法加载任何数据，跳过 {attack_display_name}")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 确定 x 轴范围（使用最长的数据）
    max_len = max(
        len(mean_attack_data) if mean_attack_data else 0,
        len(hsm_attack_data) if hsm_attack_data else 0,
        len(mean_baseline_data) if mean_baseline_data else 0,
        len(hsm_baseline_data) if hsm_baseline_data else 0
    )
    
    # 确定 display_interval
    display_interval = mean_interval or hsm_interval or mean_bl_interval or hsm_bl_interval or 1
    
    # 绘制曲线 - 使用更区分度的颜色和样式
    if mean_attack_data:
        x_axis = [r * display_interval for r in range(len(mean_attack_data))]
        ax.plot(x_axis, mean_attack_data, '-', color=COLORS['mean_attack'], marker='o', 
                label=f'Mean ({attack_display_name}, b=3)', linewidth=LINEWIDTH, 
                markersize=MARKERSIZE, markevery=max(1, len(mean_attack_data)//20),
                zorder=4, alpha=0.9)
    
    if hsm_attack_data:
        x_axis = [r * display_interval for r in range(len(hsm_attack_data))]
        ax.plot(x_axis, hsm_attack_data, '-', color=COLORS['hsm_attack'], marker='s', 
                label=f'HSM-Adaptive ({attack_display_name}, b=3)', linewidth=LINEWIDTH,
                markersize=MARKERSIZE, markevery=max(1, len(hsm_attack_data)//20),
                zorder=3, alpha=0.9)
    
    if mean_baseline_data:
        x_axis = [r * display_interval for r in range(len(mean_baseline_data))]
        ax.plot(x_axis, mean_baseline_data, '--', color=COLORS['mean_baseline'], marker='o', 
                label='Mean (Baseline, b=0)', linewidth=LINEWIDTH,
                markersize=MARKERSIZE, markevery=max(1, len(mean_baseline_data)//20), 
                alpha=0.8, zorder=2)
    
    if hsm_baseline_data:
        x_axis = [r * display_interval for r in range(len(hsm_baseline_data))]
        ax.plot(x_axis, hsm_baseline_data, '--', color=COLORS['hsm_baseline'], marker='s', 
                label='HSM-Adaptive (Baseline, b=0)', linewidth=LINEWIDTH,
                markersize=MARKERSIZE, markevery=max(1, len(hsm_baseline_data)//20), 
                alpha=0.8, zorder=1)
    
   
    ax.set_xlabel('Iterations', fontsize=FONTSIZE, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=FONTSIZE, fontweight='bold')
    ax.set_title(f'{attack_display_name} Attack Comparison (b=3, IID)', 
                 fontsize=FONTSIZE+2, fontweight='bold', pad=15)
    
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 美化图例
    legend = ax.legend(fontsize=FONTSIZE-2, loc='best', framealpha=0.95, 
                       fancybox=True, shadow=True, ncol=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    
    # 美化坐标轴
    ax.tick_params(labelsize=FONTSIZE-2, width=1.5, length=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # 设置背景色
    ax.set_facecolor('#FAFAFA')
    
    # 保存图片
    pic_dir = os.path.join(__FILE_DIR__, 'pic')
    os.makedirs(pic_dir, exist_ok=True)
    
    pic_name = f'{method}_{dataset}_{attack_name}_b3_comparison'
    png_path = os.path.join(pic_dir, f'{pic_name}.png')
    pdf_path = os.path.join(pic_dir, f'{pic_name}.pdf')
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"已保存图片: {png_path}")
    print(f"已保存图片: {pdf_path}")
    
    plt.close()

def main():
    """主函数：绘制所有攻击场景的对比图"""
    attacks = [
        ('label_flipping', 'Label Flipping'),
        ('MSA', 'MSA'),
        ('HisMSA', 'HisMSA')
    ]
    
    print("开始绘制攻击对比图...")
    print(f"数据集: {dataset}, 分区: {partition_name}")
    print(f"b=0 路径: {graph_name_b0}, b=3 路径: {graph_name_b3}\n")
    
    for attack_code, attack_display in attacks:
        print(f"\n{'='*60}")
        print(f"正在绘制 {attack_display} 攻击对比图...")
        print(f"{'='*60}")
        draw_attack_comparison(attack_code, attack_display)
    
    print("\n" + "="*60)
    print("所有图片绘制完成！")
    print("="*60)

if __name__ == '__main__':
    main()
