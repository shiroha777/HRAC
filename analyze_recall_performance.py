#!/usr/bin/env python3
"""
分析 hsm-logiid2.txt 日志中的 recall 表现
找出哪些参数导致检测失败
"""

import re
from collections import defaultdict

def parse_log_file(filepath):
    """解析日志文件，提取关键信息"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配每个迭代的统计信息
    pattern = r'\[HSM-Adaptive\] Iteration (\d+) Statistics:.*?\[Defense Metric\] Attack Detection Rate \(Recall\): ([\d.]+)% \((\d+)/(\d+)\)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    iterations = []
    for match in matches:
        iter_num = int(match.group(1))
        recall = float(match.group(2))
        detected = int(match.group(3))
        total = int(match.group(4))
        
        # 提取该迭代的详细信息
        iter_block = content[match.start():match.end()]
        
        # 提取 tau_t
        tau_match = re.search(r'tau_t \(adaptive threshold\): ([\d.-]+)', iter_block)
        tau_t = float(tau_match.group(1)) if tau_match else None
        
        # 提取权重信息
        weights_match = re.search(r'\[Adaptive Weights\].*?Global_Align: ([\d.]+).*?Self: ([\d.]+)', iter_block)
        w_a = float(weights_match.group(1)) if weights_match else None
        w_c = float(weights_match.group(2)) if weights_match else None
        
        # 提取信任分数统计
        trust_match = re.search(r'Trust Scores mean: ([\d.-]+), std: ([\d.-]+)', iter_block)
        trust_mean = float(trust_match.group(1)) if trust_match else None
        trust_std = float(trust_match.group(2)) if trust_match else None
        
        # 提取对齐分数统计
        align_match = re.search(r'Alignment mean: ([\d.-]+), std: ([\d.-]+)', iter_block)
        align_mean = float(align_match.group(1)) if align_match.group(1) else None
        align_std = float(align_match.group(2)) if align_match.group(2) else None
        
        # 提取每个客户端的信息
        client_pattern = r'(\d+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+([\d.-]+)\s+\|\s+(YES|NO)'
        clients = []
        for client_match in re.finditer(client_pattern, iter_block):
            client_id = int(client_match.group(1))
            align = float(client_match.group(2))
            self_align = float(client_match.group(3))
            mom_align = float(client_match.group(4))
            scale_dev = float(client_match.group(5))
            temporal = float(client_match.group(6))
            trust = float(client_match.group(7))
            weight = float(client_match.group(8))
            norm = float(client_match.group(9))
            is_attacker = client_match.group(11) == 'YES'
            
            clients.append({
                'id': client_id,
                'align': align,
                'self_align': self_align,
                'mom_align': mom_align,
                'scale_dev': scale_dev,
                'temporal': temporal,
                'trust': trust,
                'weight': weight,
                'norm': norm,
                'is_attacker': is_attacker
            })
        
        iterations.append({
            'iteration': iter_num,
            'recall': recall,
            'detected': detected,
            'total': total,
            'tau_t': tau_t,
            'w_a': w_a,
            'w_c': w_c,
            'trust_mean': trust_mean,
            'trust_std': trust_std,
            'align_mean': align_mean,
            'align_std': align_std,
            'clients': clients
        })
    
    return iterations

def analyze_recall_performance(iterations):
    """分析 recall 表现"""
    print("=" * 80)
    print("Recall 表现分析")
    print("=" * 80)
    
    # 统计 recall 分布
    recall_bins = {'0%': 0, '33%': 0, '67%': 0, '100%': 0}
    for it in iterations:
        if it['recall'] == 0.0:
            recall_bins['0%'] += 1
        elif it['recall'] < 50.0:
            recall_bins['33%'] += 1
        elif it['recall'] < 100.0:
            recall_bins['67%'] += 1
        else:
            recall_bins['100%'] += 1
    
    print(f"\nRecall 分布:")
    total = len(iterations)
    for bin_name, count in recall_bins.items():
        print(f"  {bin_name}: {count} 次 ({count/total*100:.1f}%)")
    
    # 分析失败案例（recall = 0%）vs 成功案例（recall = 100%）
    failed = [it for it in iterations if it['recall'] == 0.0]
    success = [it for it in iterations if it['recall'] == 100.0]
    
    print(f"\n失败案例 (recall=0%): {len(failed)} 个")
    print(f"成功案例 (recall=100%): {len(success)} 个")
    
    if len(failed) > 0 and len(success) > 0:
        print("\n" + "=" * 80)
        print("失败 vs 成功案例的参数对比")
        print("=" * 80)
        
        # 对比参数
        params = ['tau_t', 'w_a', 'w_c', 'trust_mean', 'trust_std', 'align_mean', 'align_std']
        for param in params:
            failed_vals = [it[param] for it in failed if it[param] is not None]
            success_vals = [it[param] for it in success if it[param] is not None]
            
            if failed_vals and success_vals:
                failed_mean = sum(failed_vals) / len(failed_vals)
                success_mean = sum(success_vals) / len(success_vals)
                print(f"\n{param}:")
                print(f"  失败案例均值: {failed_mean:.4f}")
                print(f"  成功案例均值: {success_mean:.4f}")
                print(f"  差异: {success_mean - failed_mean:.4f}")
        
        # 分析攻击者的特征
        print("\n" + "=" * 80)
        print("攻击者特征对比 (失败 vs 成功)")
        print("=" * 80)
        
        attacker_metrics = ['align', 'self_align', 'scale_dev', 'trust', 'weight']
        for metric in attacker_metrics:
            failed_attacker_vals = []
            success_attacker_vals = []
            
            for it in failed:
                for client in it['clients']:
                    if client['is_attacker']:
                        failed_attacker_vals.append(client[metric])
            
            for it in success:
                for client in it['clients']:
                    if client['is_attacker']:
                        success_attacker_vals.append(client[metric])
            
            if failed_attacker_vals and success_attacker_vals:
                failed_mean = sum(failed_attacker_vals) / len(failed_attacker_vals)
                success_mean = sum(success_attacker_vals) / len(success_attacker_vals)
                print(f"\n攻击者 {metric}:")
                print(f"  失败案例均值: {failed_mean:.4f}")
                print(f"  成功案例均值: {success_mean:.4f}")
                print(f"  差异: {success_mean - failed_mean:.4f}")
        
        # 找出最典型的失败案例
        print("\n" + "=" * 80)
        print("典型失败案例分析")
        print("=" * 80)
        
        # 选择几个失败案例详细分析
        for i, it in enumerate(failed[:3]):
            print(f"\n失败案例 {i+1} - Iteration {it['iteration']}:")
            print(f"  tau_t: {it['tau_t']:.4f}")
            print(f"  w_a: {it['w_a']:.4f}, w_c: {it['w_c']:.4f}")
            print(f"  Trust Scores mean: {it['trust_mean']:.4f}, std: {it['trust_std']:.4f}")
            print(f"  Alignment mean: {it['align_mean']:.4f}, std: {it['align_std']:.4f}")
            print(f"\n  攻击者详情:")
            for client in it['clients']:
                if client['is_attacker']:
                    print(f"    Client {client['id']}: align={client['align']:.4f}, "
                          f"self_align={client['self_align']:.4f}, "
                          f"scale_dev={client['scale_dev']:.4f}, "
                          f"trust={client['trust']:.4f}, weight={client['weight']:.4f}")
        
        # 找出最典型的成功案例
        print("\n" + "=" * 80)
        print("典型成功案例分析")
        print("=" * 80)
        
        for i, it in enumerate(success[:3]):
            print(f"\n成功案例 {i+1} - Iteration {it['iteration']}:")
            print(f"  tau_t: {it['tau_t']:.4f}")
            print(f"  w_a: {it['w_a']:.4f}, w_c: {it['w_c']:.4f}")
            print(f"  Trust Scores mean: {it['trust_mean']:.4f}, std: {it['trust_std']:.4f}")
            print(f"  Alignment mean: {it['align_mean']:.4f}, std: {it['align_std']:.4f}")
            print(f"\n  攻击者详情:")
            for client in it['clients']:
                if client['is_attacker']:
                    print(f"    Client {client['id']}: align={client['align']:.4f}, "
                          f"self_align={client['self_align']:.4f}, "
                          f"scale_dev={client['scale_dev']:.4f}, "
                          f"trust={client['trust']:.4f}, weight={client['weight']:.4f}")

if __name__ == '__main__':
    log_file = 'record/NeuralNetwork_cifar10/Centralized_n=10_b=3/iidPartition/hsm-logiid2.txt'
    print(f"正在分析日志文件: {log_file}")
    
    iterations = parse_log_file(log_file)
    print(f"\n解析到 {len(iterations)} 个迭代")
    
    analyze_recall_performance(iterations)
