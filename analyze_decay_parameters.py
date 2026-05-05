#!/usr/bin/env python3
"""
分析随时间衰减的参数与 recall 表现的关系
"""

import re
import numpy as np
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
            'trust_mean': trust_mean,
            'trust_std': trust_std,
            'align_mean': align_mean,
            'align_std': align_std,
            'clients': clients
        })
    
    return iterations

def analyze_decay_parameters(iterations):
    """分析衰减参数与 recall 的关系"""
    print("=" * 80)
    print("衰减参数与 Recall 表现的关系分析")
    print("=" * 80)
    
    # 按迭代阶段分组（早期、中期、后期）
    early = [it for it in iterations if it['iteration'] < 5000]
    mid = [it for it in iterations if 5000 <= it['iteration'] < 10000]
    late = [it for it in iterations if it['iteration'] >= 10000]
    
    print(f"\n迭代阶段划分:")
    print(f"  早期 (0-5000): {len(early)} 个迭代")
    print(f"  中期 (5000-10000): {len(mid)} 个迭代")
    print(f"  后期 (10000+): {len(late)} 个迭代")
    
    # 分析 tau_t 随时间的变化
    print("\n" + "=" * 80)
    print("tau_t (EMA 平滑阈值) 随时间的变化")
    print("=" * 80)
    
    for stage_name, stage_iters in [("早期", early), ("中期", mid), ("后期", late)]:
        if len(stage_iters) == 0:
            continue
        
        # 按 recall 分组
        failed = [it for it in stage_iters if it['recall'] == 0.0]
        success = [it for it in stage_iters if it['recall'] == 100.0]
        
        tau_t_vals = [it['tau_t'] for it in stage_iters if it['tau_t'] is not None]
        failed_tau_t = [it['tau_t'] for it in failed if it['tau_t'] is not None]
        success_tau_t = [it['tau_t'] for it in success if it['tau_t'] is not None]
        
        if tau_t_vals:
            print(f"\n{stage_name}阶段:")
            print(f"  所有迭代 tau_t 均值: {np.mean(tau_t_vals):.4f}, 中位数: {np.median(tau_t_vals):.4f}")
            print(f"  失败案例 (recall=0%) tau_t 均值: {np.mean(failed_tau_t):.4f}" if failed_tau_t else "  失败案例: 无")
            print(f"  成功案例 (recall=100%) tau_t 均值: {np.mean(success_tau_t):.4f}" if success_tau_t else "  成功案例: 无")
            
            # 检查 tau_t 是否随时间变得更负（EMA 累积效应）
            if len(tau_t_vals) > 10:
                early_tau = np.mean([it['tau_t'] for it in stage_iters[:len(stage_iters)//3] if it['tau_t'] is not None])
                late_tau = np.mean([it['tau_t'] for it in stage_iters[-len(stage_iters)//3:] if it['tau_t'] is not None])
                print(f"  阶段内趋势: {early_tau:.4f} -> {late_tau:.4f} (变化: {late_tau - early_tau:.4f})")
    
    # 分析 tau_t 的累积效应
    print("\n" + "=" * 80)
    print("tau_t EMA 累积效应分析")
    print("=" * 80)
    
    # 模拟 EMA 过程
    tau_ema_beta = 0.95
    print(f"\nEMA 衰减系数 beta = {tau_ema_beta}")
    print(f"这意味着新值的权重只有 {1-tau_ema_beta:.1%}，历史值的权重为 {tau_ema_beta:.1%}")
    
    # 分析失败案例中 tau_t 是否被"锁定"在负值
    failed = [it for it in iterations if it['recall'] == 0.0]
    success = [it for it in iterations if it['recall'] == 100.0]
    
    if failed and success:
        failed_tau_t = [it['tau_t'] for it in failed if it['tau_t'] is not None]
        success_tau_t = [it['tau_t'] for it in success if it['tau_t'] is not None]
        
        print(f"\n失败案例 tau_t 分布:")
        print(f"  均值: {np.mean(failed_tau_t):.4f}")
        print(f"  中位数: {np.median(failed_tau_t):.4f}")
        print(f"  最小值: {np.min(failed_tau_t):.4f}")
        print(f"  最大值: {np.max(failed_tau_t):.4f}")
        print(f"  标准差: {np.std(failed_tau_t):.4f}")
        
        print(f"\n成功案例 tau_t 分布:")
        print(f"  均值: {np.mean(success_tau_t):.4f}")
        print(f"  中位数: {np.median(success_tau_t):.4f}")
        print(f"  最小值: {np.min(success_tau_t):.4f}")
        print(f"  最大值: {np.max(success_tau_t):.4f}")
        print(f"  标准差: {np.std(success_tau_t):.4f}")
        
        # 检查是否有"锁定"现象（tau_t 持续为负且变化很小）
        print(f"\n[问题诊断]:")
        if np.mean(failed_tau_t) < -5.0:
            print(f"  失败案例的 tau_t 均值 ({np.mean(failed_tau_t):.4f}) 非常负")
            print(f"  这可能是因为 EMA 累积了早期的负值，导致阈值被'锁定'在负值")
            print(f"  即使后续信任分数改善，EMA 也需要很长时间才能恢复")
    
    # 分析信任分数方差随时间的变化
    print("\n" + "=" * 80)
    print("信任分数方差随时间的变化（影响 tau_t 计算）")
    print("=" * 80)
    
    for stage_name, stage_iters in [("早期", early), ("中期", mid), ("后期", late)]:
        if len(stage_iters) == 0:
            continue
        
        trust_stds = [it['trust_std'] for it in stage_iters if it['trust_std'] is not None]
        failed_trust_stds = [it['trust_std'] for it in stage_iters if it['recall'] == 0.0 and it['trust_std'] is not None]
        success_trust_stds = [it['trust_std'] for it in stage_iters if it['recall'] == 100.0 and it['trust_std'] is not None]
        
        if trust_stds:
            print(f"\n{stage_name}阶段:")
            print(f"  所有迭代 trust_std 均值: {np.mean(trust_stds):.4f}")
            print(f"  失败案例 trust_std 均值: {np.mean(failed_trust_stds):.4f}" if failed_trust_stds else "  失败案例: 无")
            print(f"  成功案例 trust_std 均值: {np.mean(success_trust_stds):.4f}" if success_trust_stds else "  成功案例: 无")
    
    # 分析 rho 参数的影响（历史对齐的衰减）
    print("\n" + "=" * 80)
    print("历史对齐衰减参数 (rho) 的影响分析")
    print("=" * 80)
    print("\n代码中使用 rho=0.9 来衰减历史对齐信息")
    print("这意味着:")
    print("  - 当前轮的对齐权重: 10%")
    print("  - 历史累积的对齐权重: 90%")
    print("  - 这会导致历史信息'主导'当前判断")
    
    # 检查攻击者的对齐分数是否随时间改善（因为历史累积）
    if failed:
        print("\n检查失败案例中攻击者的对齐分数模式:")
        early_failed = [it for it in failed if it['iteration'] < 5000]
        late_failed = [it for it in failed if it['iteration'] >= 10000]
        
        if early_failed and late_failed:
            early_attacker_aligns = []
            late_attacker_aligns = []
            
            for it in early_failed:
                for client in it['clients']:
                    if client['is_attacker']:
                        early_attacker_aligns.append(client['align'])
            
            for it in late_failed:
                for client in it['clients']:
                    if client['is_attacker']:
                        late_attacker_aligns.append(client['align'])
            
            if early_attacker_aligns and late_attacker_aligns:
                print(f"  早期失败案例中攻击者对齐分数均值: {np.mean(early_attacker_aligns):.4f}")
                print(f"  后期失败案例中攻击者对齐分数均值: {np.mean(late_attacker_aligns):.4f}")
                print(f"  变化: {np.mean(late_attacker_aligns) - np.mean(early_attacker_aligns):.4f}")
    
    # 总结和建议
    print("\n" + "=" * 80)
    print("总结和建议")
    print("=" * 80)
    
    print("\n[关键发现]:")
    print("1. tau_t 使用 EMA (beta=0.95) 平滑，这意味着:")
    print("   - 如果早期出现负值，会持续影响后续阈值")
    print("   - 需要很长时间才能从负值恢复")
    print("   - 这可能导致阈值被'锁定'在负值")
    
    print("\n2. 历史对齐使用 rho=0.9 衰减，这意味着:")
    print("   - 历史信息权重很高 (90%)")
    print("   - 如果攻击者早期获得正对齐，历史会持续影响")
    print("   - 即使当前轮对齐为负，历史仍可能主导")
    
    print("\n[改进建议]:")
    print("1. 对 tau_t 进行 clamp，防止过度负向:")
    print("   tau_t = torch.clamp(tau_t, min=-1.0, max=0.0)")
    
    print("\n2. 降低 tau_ema_beta，使阈值更敏感:")
    print("   tau_ema_beta = 0.8 或 0.85 (而不是 0.95)")
    
    print("\n3. 降低历史对齐的权重 (rho):")
    print("   rho = 0.7 或 0.8 (而不是 0.9)，使当前信息更重要")
    
    print("\n4. 引入'重置机制':")
    print("   当检测到异常时，重置相关历史信息")

if __name__ == '__main__':
    log_file = 'record/NeuralNetwork_cifar10/Centralized_n=10_b=3/iidPartition/hsm-logiid2.txt'
    print(f"正在分析日志文件: {log_file}")
    
    iterations = parse_log_file(log_file)
    print(f"\n解析到 {len(iterations)} 个迭代")
    
    analyze_decay_parameters(iterations)
