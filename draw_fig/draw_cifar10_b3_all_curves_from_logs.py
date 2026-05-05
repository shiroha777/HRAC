"""
CIFAR10 b=3 + Baseline b=0: 从 cache 加载 acc_path，绘制 7×2 大图。
- 数据来源：
  - b=0: record/NeuralNetwork_cifar10/Centralized_n=10_b=0/{iidPartition|LabelSeperation} → 第一行「Baseline (b=0)」各 aggregator
  - b=3: record/NeuralNetwork_cifar10/Centralized_n=10_b=3/{iidPartition|LabelSeperation} → 后 6 行各攻击
- 行：Baseline (b=0), label_flipping, MSA, HisMSA, bit_flipping, ALIE, IPM（不含空的 baseline b=3 行）
- 列：IID (iidPartition), Non-IID (LabelSeperation)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(__FILE_DIR__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path, isfile_in_cache

RECORD_DIR = os.path.join(project_root, 'record')
TASK_GRAPH = 'NeuralNetwork_cifar10'
GRAPH_NAME = 'Centralized_n=10_b=3'
GRAPH_NAME_B0 = 'Centralized_n=10_b=0'  # 无攻击时各 aggregator 表现，画在第一行
PARTITIONS = [
    ('iid', 'iidPartition'),
    ('noniid', 'LabelSeperation'),
]
set_cache_path(RECORD_DIR)

# 行顺序（与图中一致）；第一行为 b=0 无攻击
ROW_B0_LABEL = 'Baseline (b=0)'
ATTACK_ORDER = [ROW_B0_LABEL, 'label_flipping', 'MSA', 'HisMSA', 'bit_flipping', 'ALIE', 'IPM']
# 文件名前缀：CMomentum_{attack}，用于匹配 cache 文件名（长前缀优先，故 MSA_MSA / HisMSA_HisMSA 放前）
ATTACK_FILE_PREFIX = {
    'MSA': 'CMomentum_MSA_MSA',
    'HisMSA': 'CMomentum_HisMSA_HisMSA',
    'baseline': 'CMomentum_baseline',
    'label_flipping': 'CMomentum_label_flipping',
    'bit_flipping': 'CMomentum_bit_flipping',
    'ALIE': 'CMomentum_ALIE',
    'IPM': 'CMomentum_IPM',
}
# 聚合器 key（与 main 里 aggregation.name 一致）-> 图例显示名
AGG_DISPLAY = {
    'mean': 'Mean',
    'trimmed_mean': 'Trimmed Mean',
    'faba': 'FABA',
    'CC_tau=0.1': 'CC(τ=0.1)',
    'CC_tau=0.3': 'CC(τ=0.3)',
    'CC_tau=100': 'CC(τ=100)',
    'LFighter': 'LFighter',
    'HRAC': 'HRAC',
    'HSM-Lite': 'HSM',
    'HSM': 'HSM',
}
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))
# HRAC=紫，CC(τ=100) 用橙色区分，避免与 HRAC 混淆
METHOD_COLORS = {
    'Mean': COLORS[0],
    'Trimmed Mean': COLORS[1],
    'FABA': COLORS[2],
    'CC(τ=0.1)': COLORS[3],
    'CC(τ=0.3)': COLORS[3],
    'CC(τ=100)': (0.9, 0.45, 0.0),   # 橙色，与 HRAC 紫色区分
    'LFighter': COLORS[5],
    'HRAC': COLORS[4],
    'HSM': COLORS[7],
    'HSM-Adaptive': COLORS[6],
}
PREFERRED_METHOD_ORDER = ['Mean', 'FABA', 'CC(τ=0.1)', 'CC(τ=0.3)', 'LFighter', 'HRAC', 'HSM', 'HSM-Adaptive', 'Trimmed Mean']


def convert_to_list(data):
    """将 record 中的 acc_path 转为 list of float。"""
    if data is None:
        return None
    try:
        import torch
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy() if data.is_cuda else data.detach().numpy()
    except ImportError:
        pass
    if isinstance(data, np.ndarray):
        data = data.tolist()
    if isinstance(data, list):
        data = [float(x) if not isinstance(x, (list, tuple)) else float(x[0]) for x in data]
    return data


def parse_cache_filename(fname):
    """
    从 cache 文件名解析 (attack_key, agg_key)。
    文件名格式：CMomentum_{attack}_{aggregation}，如 CMomentum_ALIE_HRAC、CMomentum_label_flipping_HRAC。
    长前缀优先匹配（保证 MSA_MSA、HisMSA_HisMSA 先于单段）。
    """
    if not fname or fname.endswith('.txt') or fname.endswith('.md'):
        return None, None
    # 按前缀长度降序，优先匹配最长
    for attack_key in sorted(ATTACK_FILE_PREFIX.keys(), key=lambda k: -len(ATTACK_FILE_PREFIX[k])):
        prefix = ATTACK_FILE_PREFIX[attack_key]
        if fname.startswith(prefix + '_'):
            agg_key = fname[len(prefix) + 1:]
            return attack_key, agg_key
    return None, None


def collect_from_cache():
    """
    扫描 b=0 与 b=3 下 iidPartition / LabelSeperation 的 cache，
    返回 data[(row_label, partition)][method_display] = (iters, accs)。
    b=0 的数据统一归为 row_label=ROW_B0_LABEL（第一行）；b=3 用 attack_key 作为 row_label。
    """
    data = {}
    # (graph_name, row_label): row_label=None 表示用 parse 出的 attack_key 作为行键
    for graph_name, row_label in [(GRAPH_NAME_B0, ROW_B0_LABEL), (GRAPH_NAME, None)]:
        path_list_base = [TASK_GRAPH, graph_name]
        for part_key, part_dir in PARTITIONS:
            path_list = path_list_base + [part_dir]
            full_dir = os.path.join(RECORD_DIR, *path_list)
            if not os.path.isdir(full_dir):
                continue
            for fname in os.listdir(full_dir):
                if fname.endswith('.txt') or fname.endswith('.md'):
                    continue
                attack_key, agg_key = parse_cache_filename(fname)
                if attack_key is None:
                    continue
                try:
                    if not isfile_in_cache(fname, path_list):
                        continue
                    record = load_file_in_cache(fname, path_list)
                except Exception:
                    continue
                if not isinstance(record, dict) or 'acc_path' not in record:
                    continue
                acc_path = convert_to_list(record.get('acc_path'))
                if not acc_path:
                    continue
                interval = record.get('display_interval', 100)
                iters = [k * interval for k in range(len(acc_path))]
                accs = acc_path
                method_display = AGG_DISPLAY.get(agg_key, agg_key)
                key = (row_label if row_label is not None else attack_key, part_key)
                if key not in data:
                    data[key] = {}
                data[key][method_display] = (iters, accs)
    return data


def draw_all_curves(data, save_dir):
    """绘制 7×2 大图：第 1 行 Baseline (b=0)，第 2–7 行各攻击；每子图内画所有有数据的曲线。"""
    n_rows = len(ATTACK_ORDER)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows), sharex=True)
    part_titles = {'iid': 'IID', 'noniid': 'Non-IID'}
    for i, row_label in enumerate(ATTACK_ORDER):
        for j, (part_key, _) in enumerate(PARTITIONS):
            ax = axes[i, j]
            key = (row_label, part_key)
            curves = data.get(key, {})
            order = [m for m in PREFERRED_METHOD_ORDER if m in curves]
            order += sorted(k for k in curves if k not in order)
            for idx, method in enumerate(order):
                iters, accs = curves[method]
                color = METHOD_COLORS.get(method, COLORS[idx % len(COLORS)])
                ax.plot(iters, accs, label=method, color=color, linewidth=2, alpha=0.9)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(0, 20000)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#fafafa')
            ax.set_title(f'{row_label} | {part_titles.get(part_key, part_key)}', fontsize=11)
            if i == n_rows - 1:
                ax.set_xlabel('Iterations', fontsize=11)
            if curves:
                ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(save_dir, 'CMomentum_cifar10_b3_all_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    try:
        plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    except Exception:
        pass
    plt.close()
    print(f'Saved: {out_path}')
    return out_path


def main():
    save_dir = os.path.join(__FILE_DIR__, 'pic')
    os.makedirs(save_dir, exist_ok=True)
    print('Loading b=0 and b=3 data from cache (CMomentum_* files).')
    for graph_name, label in [(GRAPH_NAME_B0, 'b=0'), (GRAPH_NAME, 'b=3')]:
        base = os.path.join(RECORD_DIR, TASK_GRAPH, graph_name)
        for part_key, part_dir in PARTITIONS:
            dirpath = os.path.join(base, part_dir)
            if os.path.isdir(dirpath):
                cache_files = [f for f in os.listdir(dirpath) if not f.endswith('.txt') and not f.endswith('.md')]
                print(f'  {label} {part_dir} ({part_key}): {len(cache_files)} cache files')
    data = collect_from_cache()
    if not data:
        print('No data found. Run experiments to generate CMomentum_* cache under Centralized_n=10_b=0 and b=3.')
        return
    for (row_label, part), curves in sorted(data.items()):
        print(f'  {row_label} | {part}: {list(curves.keys())}')
    print('Drawing 7×2 figure (row0=Baseline b=0, rows1-6=attacks b=3)...')
    draw_all_curves(data, save_dir)
    print('Done.')


if __name__ == '__main__':
    main()
