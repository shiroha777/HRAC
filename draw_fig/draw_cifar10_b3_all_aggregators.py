"""
CIFAR10, b=3: 加载各攻击 × (IID/Non-IID) × 聚合器的记录，打印 acc 表并绘制曲线对比图。
- 表格：不同攻击、IID/Non-IID 下各聚合器的表现（最终 acc / max acc）
- 图片：全部为曲线对比——每个 (攻击, 分区) 一张曲线图 + 一张总图包含所有子图曲线
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

__FILE_DIR__ = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(__FILE_DIR__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ByrdLab.library.cache_io import load_file_in_cache, set_cache_path, isfile_in_cache

__CACHE_DIR__ = 'record'
__CACHE_PATH__ = os.path.join(project_root, __CACHE_DIR__)
set_cache_path(__CACHE_PATH__)

# 配置
METHOD = 'CMomentum'
TASK_NAME = 'NeuralNetwork'
DATASET = 'cifar10'
GRAPH_NAME = 'Centralized_n=10_b=3'
PARTITIONS = [
    ('iid', 'iidPartition'),
    ('noniid', 'LabelSeperation'),
]
# 攻击在文件名中的形式: (env.name, attack_name) -> file prefix
# CMomentum_baseline_agg, CMomentum_label_flipping_agg, CMomentum_MSA_MSA_agg, CMomentum_HisMSA_HisMSA_agg
ATTACK_FILE_PREFIX = {
    'baseline': 'CMomentum_baseline',
    'label_flipping': 'CMomentum_label_flipping',
    'MSA': 'CMomentum_MSA_MSA',
    'HisMSA': 'CMomentum_HisMSA_HisMSA',
}
# 文件名中聚合器名：与 main 里 aggregation.name 一致
PREFERRED_ORDER = ['mean', 'trimmed_mean', 'faba', 'CC_tau=0.1', 'CC_tau=0.3', 'LFighter', 'HRAC', 'HSM-Lite']
AGG_DISPLAY = {
    'mean': 'Mean',
    'trimmed_mean': 'Trimmed Mean',
    'faba': 'FABA',
    'CC_tau=0.1': 'CC(τ=0.1)',
    'CC_tau=0.3': 'CC(τ=0.3)',
    'LFighter': 'LFighter',
    'HRAC': 'HRAC',
    'HSM-Lite': 'HSM',
}

COLORS = plt.cm.tab10(np.linspace(0, 1, max(len(PREFERRED_ORDER), 10)))
AGG_COLOR = {agg: COLORS[i % len(COLORS)] for i, agg in enumerate(PREFERRED_ORDER)}

def convert_to_list(data):
    import torch
    import numpy as np
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy() if data.is_cuda else data.detach().numpy()
    if isinstance(data, np.ndarray):
        data = data.tolist()
    if isinstance(data, list):
        data = [float(x) if not isinstance(x, (list, tuple)) else float(x[0]) for x in data]
    return data

def try_load_record(file_name, path_list):
    try:
        if isfile_in_cache(file_name, path_list):
            return load_file_in_cache(file_name, path_list)
    except Exception as e:
        print(f"  load fail {file_name}: {e}")
    return None

def guess_aggregator_from_filename(file_name):
    """从文件名 CMomentum_*_<aggregation> 解析出 aggregation 部分"""
    for prefix in ATTACK_FILE_PREFIX.values():
        if file_name.startswith(prefix + '_'):
            return file_name[len(prefix) + 1:]
    return None

def collect_all_results():
    """扫描 record 目录，收集所有 (attack, partition) -> { agg -> record }"""
    taskname = f'{TASK_NAME}_{DATASET}'
    results = {}
    for part_key, part_dir in PARTITIONS:
        path_list = [taskname, GRAPH_NAME, part_dir]
        full_dir = os.path.join(__CACHE_PATH__, *path_list)
        if not os.path.isdir(full_dir):
            continue
        for file_name in os.listdir(full_dir):
            if file_name.endswith('.txt'):
                continue
            # 解析攻击类型
            attack_key = None
            for ak, prefix in ATTACK_FILE_PREFIX.items():
                if file_name.startswith(prefix + '_'):
                    attack_key = ak
                    break
            if attack_key is None:
                continue
            agg_key = guess_aggregator_from_filename(file_name)
            if agg_key is None:
                agg_key = file_name  # 整段作 key
            key = (attack_key, part_key)
            if key not in results:
                results[key] = {}
            record = try_load_record(file_name, path_list)
            if record and isinstance(record, dict) and 'acc_path' in record:
                results[key][agg_key] = {
                    'record': record,
                    'file_name': file_name,
                }
    return results

def get_acc_stats(record):
    acc_path = convert_to_list(record.get('acc_path'))
    if not acc_path:
        return None, None, None, None
    display_interval = record.get('display_interval', 100)
    final_acc = float(acc_path[-1]) if acc_path else None
    max_acc = max(acc_path) if acc_path else None
    return acc_path, display_interval, final_acc, max_acc

def ordered_aggregators(agg_data):
    """返回聚合器列表：优先顺序 + 其余按 key 排序"""
    seen = set()
    out = []
    for agg in PREFERRED_ORDER:
        if agg in agg_data:
            seen.add(agg)
            out.append(agg)
    for agg in sorted(agg_data.keys()):
        if agg not in seen:
            out.append(agg)
    return out

def print_table(results):
    """打印表格：攻击 × 分区 × 聚合器 -> 最终 acc"""
    print('\n' + '='*80)
    print('CIFAR10, b=3 — 各攻击与 IID/Non-IID 下各聚合器最终准确率 (Final Acc)')
    print('='*80)
    for (attack_key, part_key) in sorted(results.keys()):
        print(f'\n--- Attack: {attack_key}  |  Partition: {part_key} ---')
        agg_data = results[(attack_key, part_key)]
        for agg in ordered_aggregators(agg_data):
            rec = agg_data.get(agg)
            if rec:
                _, _, final_acc, max_acc = get_acc_stats(rec['record'])
                print(f"  {AGG_DISPLAY.get(agg, agg):18s}: final={final_acc:.4f}  max={max_acc:.4f}")
            else:
                print(f"  {AGG_DISPLAY.get(agg, agg):18s}: (no data)")
    print('\n' + '='*80)

def draw_curves_one(results, attack_key, part_key, save_dir):
    """绘制单张图：某一 (attack, partition) 下各聚合器的 acc 曲线"""
    key = (attack_key, part_key)
    if key not in results or not results[key]:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    agg_data = results[key]
    for agg in ordered_aggregators(agg_data):
        rec = agg_data.get(agg)
        if not rec:
            continue
        acc_path, display_interval, _, _ = get_acc_stats(rec['record'])
        if not acc_path:
            continue
        interval = display_interval or 100
        x = [i * interval for i in range(len(acc_path))]
        label = AGG_DISPLAY.get(agg, agg)
        color = AGG_COLOR.get(agg, 'gray')
        ax.plot(x, acc_path, label=label, color=color, linewidth=2, alpha=0.9)
    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    part_title = 'IID' if part_key == 'iid' else 'Non-IID'
    ax.set_title(f'CIFAR10 b=3 — {attack_key} under {part_title}', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')
    plt.tight_layout()
    fname = f'CMomentum_cifar10_b3_{attack_key}_{part_key}_aggregators.png'
    out_path = os.path.join(save_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')

def draw_all_curves_subplots(results, save_dir):
    """一张大图：所有 (攻击 × 分区) 子图，每个子图内为各聚合器的 acc 曲线"""
    attacks = ['baseline', 'label_flipping', 'MSA', 'HisMSA']
    parts = ['iid', 'noniid']
    n_attacks, n_parts = len(attacks), len(parts)
    fig, axes = plt.subplots(n_attacks, n_parts, figsize=(6 * n_parts, 5 * n_attacks))
    if n_attacks == 1 and n_parts == 1:
        axes = [[axes]]
    elif n_attacks == 1:
        axes = [axes]
    elif n_parts == 1:
        axes = [[a] for a in axes]
    for i, attack_key in enumerate(attacks):
        for j, part_key in enumerate(parts):
            ax = axes[i][j]
            key = (attack_key, part_key)
            agg_data = results.get(key, {})
            for agg in ordered_aggregators(agg_data):
                rec = agg_data.get(agg)
                if not rec:
                    continue
                acc_path, display_interval, _, _ = get_acc_stats(rec['record'])
                if not acc_path:
                    continue
                interval = display_interval or 100
                x = [k * interval for k in range(len(acc_path))]
                label = AGG_DISPLAY.get(agg, agg)
                color = AGG_COLOR.get(agg, 'gray')
                ax.plot(x, acc_path, label=label, color=color, linewidth=2, alpha=0.9)
            ax.set_xlabel('Iterations', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            part_title = 'IID' if part_key == 'iid' else 'Non-IID'
            ax.set_title(f'{attack_key} | {part_title}', fontsize=12)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#fafafa')
            ax.set_ylim(0, 1.0)
    plt.tight_layout()
    out_path = os.path.join(save_dir, 'CMomentum_cifar10_b3_all_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f'  Saved all curves (subplots): {out_path}')

def main():
    save_dir = os.path.join(__FILE_DIR__, 'pic')
    os.makedirs(save_dir, exist_ok=True)
    print('Loading records from:', __CACHE_PATH__)
    results = collect_all_results()
    if not results:
        print('No records found. Run run_cifar10_b3_all.py first to generate data.')
        return
    print_table(results)
    print('\nDrawing curve figures (one per attack × partition)...')
    for (attack_key, part_key) in sorted(results.keys()):
        draw_curves_one(results, attack_key, part_key, save_dir)
    print('Drawing one figure with all curves (subplots)...')
    draw_all_curves_subplots(results, save_dir)
    print('Done.')

if __name__ == '__main__':
    main()
