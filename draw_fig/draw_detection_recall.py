"""
自动绘制攻击者检测 recall 曲线
从训练记录中读取 detection_recall_path 并绘制图像
"""
import matplotlib.pyplot as plt
import os
import sys
import pickle

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

def draw_detection_recall_from_record(record_file_path=None, title=None, path_list=None):
    """
    从记录文件中读取 detection_recall_path 并绘制曲线
    
    Args:
        record_file_path: 记录文件的完整路径（可选，如果提供则直接加载）
        title: 记录标题（用于从缓存加载）
        path_list: 缓存路径列表，格式如 ['NeuralNetwork_cifar10', 'Centralized_n=10_b=3', 'iidPartition']
    """
    record = None
    
    # 如果提供了文件路径，直接加载
    if record_file_path and os.path.exists(record_file_path):
        try:
            with open(record_file_path, 'rb') as f:
                record = pickle.load(f)
        except Exception as e:
            print(f"无法从文件路径加载: {e}")
    
    # 如果从缓存加载
    if record is None and title:
        if path_list:
            # 使用提供的路径
            try:
                record = load_file_in_cache(title, path_list)
            except:
                pass
        else:
            # 尝试常见的路径结构
            possible_paths = [
                ['NeuralNetwork_cifar10', 'Centralized_n=10_b=3', 'iidPartition'],
                ['NeuralNetwork_cifar10', 'Centralized_n=10_b=3', 'LabelSeperation'],
                ['NeuralNetwork_cifar10', 'Centralized_n=10_b=0', 'LabelSeperation'],
                ['NeuralNetwork_mnist', 'Centralized_n=10_b=3', 'iidPartition'],
                ['NeuralNetwork_mnist', 'Centralized_n=10_b=3', 'LabelSeperation'],
            ]
            
            for path_list_try in possible_paths:
                try:
                    record = load_file_in_cache(title, path_list_try)
                    if record and 'detection_recall_path' in record:
                        break
                except:
                    continue
    
    if record is None:
        print("错误: 无法加载记录文件")
        if title:
            print(f"尝试的标题: {title}")
        if record_file_path:
            print(f"尝试的文件路径: {record_file_path}")
        return
    
    if record is None:
        print("错误: 无法加载记录文件")
        return
    
    if 'detection_recall_path' not in record:
        print("警告: 记录文件中没有 detection_recall_path 数据")
        return
    
    # 提取数据
    recall_path = convert_to_list(record.get('detection_recall_path', []))
    precision_path = convert_to_list(record.get('detection_precision_path', []))
    
    if not recall_path:
        print("警告: detection_recall_path 为空")
        return
    
    # 获取迭代信息
    display_interval = record.get('display_interval', 100)
    log_interval = record.get('log_interval', 100)  # 从记录中获取 log_interval
    total_iterations = record.get('total_iterations', len(recall_path) * log_interval)
    
    # 生成 x 轴（iterations）- 基于 log_interval
    # recall_path 是按 log_interval 记录的，第一个点通常在 iteration 0
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
    
    # 生成图片文件名
    if title:
        pic_name = f'{title}_detection_recall'
    else:
        pic_name = 'detection_recall'
    
    pic_path = os.path.join(pic_dir, pic_name + '.png')
    plt.savefig(pic_path, dpi=300, bbox_inches='tight')
    print(f"检测 recall 曲线已保存到: {pic_path}")
    
    # 同时保存 PDF 格式
    pdf_path = os.path.join(pic_dir, pic_name + '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"PDF 已保存到: {pdf_path}")
    
    # 显示图片
    plt.show()

if __name__ == '__main__':
    # 从命令行参数获取文件路径或标题
    if len(sys.argv) >= 2:
        arg1 = sys.argv[1]
        # 判断是文件路径还是标题
        if os.path.exists(arg1):
            record_file_path = arg1
            title = sys.argv[2] if len(sys.argv) >= 3 else None
            path_list = None
            if len(sys.argv) >= 4:
                # 如果提供了路径列表（以逗号分隔）
                path_list = sys.argv[3].split(',')
            draw_detection_recall_from_record(record_file_path, title, path_list)
        else:
            # 假设是标题，从缓存加载
            title = arg1
            path_list = None
            if len(sys.argv) >= 3:
                # 如果提供了路径列表（以逗号分隔）
                path_list = sys.argv[2].split(',')
            draw_detection_recall_from_record(None, title, path_list)
    else:
        print("用法: python draw_detection_recall.py <record_file_path> [title] [path_list]")
        print("或者: python draw_detection_recall.py <title> [path_list] (从缓存加载)")
        print("示例: python draw_detection_recall.py CMomentum_label_flipping_HSM-Adaptive")
