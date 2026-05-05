"""
在 CIFAR10 下、b=3 时，运行所有组合：攻击 × (IID/Non-IID) × 聚合器
用于后续绘制各聚合器在不同攻击与数据分布下的 acc 表现。
"""
import subprocess
import sys
import os

# 攻击: none -> baseline, label_flipping, msa, hismsa
ATTACKS = [
    ('none', 'baseline'),
    ('label_flipping', 'label_flipping'),
    ('msa', 'MSA'),
    ('hismsa', 'HisMSA'),
]

# 数据划分: iid, noniid
PARTITIONS = [
    ('iid', 'iidPartition'),
    ('noniid', 'LabelSeperation'),
]

# 聚合器 (与 main CMomentum.py 的 --aggregation 一致)
AGGREGATORS = [
    'mean',
    'trimmed-mean',
    'faba',
    'cc',
    'lfighter',
    'hrac',
    'hsm',
]

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(project_root, 'main CMomentum.py')
    if not os.path.isfile(main_script):
        print(f'Error: main script not found: {main_script}')
        sys.exit(1)

    total = len(ATTACKS) * len(PARTITIONS) * len(AGGREGATORS)
    current = 0
    for attack_arg, _ in ATTACKS:
        for part_arg, _ in PARTITIONS:
            for agg in AGGREGATORS:
                current += 1
                cmd = [
                    sys.executable,
                    main_script,
                    '--attack', attack_arg,
                    '--aggregation', agg,
                    '--data-partition', part_arg,
                ]
                print(f'\n[{current}/{total}] python main CMomentum.py --attack {attack_arg} --aggregation {agg} --data-partition {part_arg}')
                sys.stdout.flush()
                ret = subprocess.run(cmd, cwd=project_root)
                if ret.returncode != 0:
                    print(f'Warning: command failed with code {ret.returncode}')
    print(f'\nDone. Total runs: {total}')

if __name__ == '__main__':
    main()
