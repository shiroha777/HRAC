from argsParser import args

from ByrdLab import FEATURE_TYPE
from ByrdLab.aggregation import C_mean, C_trimmed_mean, C_faba, C_centered_clipping, C_LFighter, C_HSM_FedAvg, C_HRAC
from ByrdLab.attack import C_gaussian, C_same_value, C_sign_flipping, C_bit_flipping, C_IPM, C_ALIE, C_TauBoundary, feature_label_random, \
                            label_flipping, label_random, furthest_label_flipping, adversarial_label_flipping, feature_label_random, HisMSA, MSA
from ByrdLab.centraliedAlgorithm import CSGD, CSGD_under_DPA, CMomentum_under_DPA, CMomentum_with_LFighter_under_DPA, CMomentum_under_HisMSA
from ByrdLab.library.cache_io import dump_file_in_cache, load_file_in_cache
from ByrdLab.library.dataset import ijcnn, mnist, fashionmnist, cifar10, mnist_sorted_by_labels
from ByrdLab.library.learnRateController import ladder_lr, one_over_sqrt_k_lr, twoStage_lr
from ByrdLab.library.partition import (LabelSeperation, TrivalPartition,
                                   iidPartition, DirichletIiiPartition, DirichletMildPartition, DirichletNoniidPartition,
                                    DirichletPartition_a, DirichletPartition_b, DirichletPartition_c, DirichletPartition_d, DirichletPartition_e, DirichletPartition_f)
from ByrdLab.library.tool import log
from ByrdLab.tasks.logisticRegression import LogisticRegressionTask
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask
from ByrdLab.tasks.leastSquare import LeastSquareToySet, LeastSquareToyTask
from ByrdLab.tasks.neuralNetwork import NeuralNetworkTask

node_size = 10
byzantine_size = 3  # 增加到3个恶意节点（30%的比例，更明显的攻击效果）

all_nodes = list(range(node_size))
honest_nodes = list(range(node_size - byzantine_size))
byzantine_nodes = [node for node in all_nodes if node not in honest_nodes]

args.lr_ctrl = 'constant'

# run for centralized algorithm
# ===========================================

# -------------------------------------------
# define learning task
# -------------------------------------------
# data_package = mnist()
# task = softmaxRegressionTask(data_package, batch_size=32)

data_package = cifar10()
task = NeuralNetworkTask(data_package, batch_size=32)
task.super_params['lr'] = 0.01

# data_package = mnist()
# task = NeuralNetworkTask(data_package, batch_size=32)

# ===========================================

# -------------------------------------------
# define attack
# -------------------------------------------
if args.attack == 'none':
    attack = None
elif args.attack == 'label_flipping':
    attack = label_flipping()
elif args.attack == 'label_random':
    attack = label_random()
elif args.attack == 'feature_label_random':
    attack = feature_label_random()
elif args.attack == 'furthest_label_flipping':
    attack = furthest_label_flipping()
elif args.attack == 'hismsa':
    attack = HisMSA(
        shuffle_prob=1.0,                    # Apply attack every round
        scaling_factor_range=(0.1, 10.0),    # Very large scaling range for STRONG attack (same as MSA)
        strict_equivalence=False,            # Allow inconsistency for stronger attack
        warmup_rounds=3                     # Reduced warmup for faster boundary adaptation
    )
elif args.attack == 'msa':
    attack = MSA(
        shuffle_prob=1.0,                    # Apply shuffle attack every round
        scaling_factor_range=(0.1, 10.0),    # Very large scaling range for STRONG attack
        strict_equivalence=False,            # Allow inconsistency for stronger attack
        break_equivalence=True,              # BREAK equivalence to create actual attack effect (α and 1/α do NOT cancel)
        equivalence_break_ratio=1.0          # 100% of layers break equivalence for MAXIMUM attack strength
    )
elif args.attack == 'bf' or args.attack == 'bit_flipping':
    # Bit Flipping: Byzantine sends -mean(honest). From byzantine-robust-optimizer.
    attack = C_bit_flipping(honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)
elif args.attack == 'ipm':
    # Inner Product Manipulation: malicious = -epsilon * mean(honest). From byzantine-robust-optimizer.
    attack = C_IPM(honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes, epsilon=0.1)
elif args.attack == 'alie':
    # A Little Is Enough. From byzantine-robust-optimizer.
    attack = C_ALIE(honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)
elif args.attack == 'tau_boundary':
    # Tau-Boundary (strong): round-gradient full-information, sees honest updates this round. Pressure test.
    attack = C_TauBoundary(honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes, use_honest=True)
elif args.attack == 'tau_boundary_weak':
    # Tau-Boundary (realistic): only own gradient, no honest_nodes/honest_stack. Cross-device FL style.
    attack = C_TauBoundary(honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes, use_honest=False)
if args.attack == 'none':
    attack_name = 'baseline'
    byzantine_size = 0

    honest_nodes = list(range(node_size))
    byzantine_nodes = []
else:
    attack_name = attack.name


# -------------------------------------------
# define learning rate control rule
# -------------------------------------------
if args.lr_ctrl == 'constant':
    lr_ctrl = None
elif args.lr_ctrl == '1/sqrt k':
    lr_ctrl = one_over_sqrt_k_lr(a=1, b=1)
    # super_params = task.super_params
    # total_iterations = super_params['rounds']*super_params['display_interval']
    # lr_ctrl = one_over_sqrt_k_lr(total_iteration=total_iterations,
    #                              a=math.sqrt(1001), b=1000)
elif args.lr_ctrl == 'ladder':
    decreasing_iter_ls = [5000, 10000, 15000]
    proportion_ls = [0.3, 0.2, 0.1]
    lr_ctrl = ladder_lr(decreasing_iter_ls, proportion_ls)
else:
    assert False, 'unknown lr-ctrl'

# Two-stage learning rate:
#   - for iteration < 500: 0.01
#   - for iteration >= 500: 0.02
lr_ctrl = twoStage_lr(switch_iteration=500, lr1=0.01, lr2=0.01)

# ===========================================
    
    
# -------------------------------------------
# define data partition
# -------------------------------------------
if args.data_partition == 'trival':
    partition_cls = TrivalPartition
elif args.data_partition == 'iid':
    partition_cls = iidPartition
elif args.data_partition == 'noniid':
    partition_cls = LabelSeperation
elif args.data_partition == 'dirichlet_iid':
    partition_cls = DirichletIiiPartition
elif args.data_partition == 'dirichlet_mild':
    partition_cls = DirichletMildPartition
elif args.data_partition == 'dirichlet_noniid':
    partition_cls = DirichletNoniidPartition
elif args.data_partition == 'dirichlet_a':
    partition_cls = DirichletPartition_a
elif args.data_partition == 'dirichlet_b':
    partition_cls = DirichletPartition_b
elif args.data_partition == 'dirichlet_c':
    partition_cls = DirichletPartition_c
elif args.data_partition == 'dirichlet_d':
    partition_cls = DirichletPartition_d
elif args.data_partition == 'dirichlet_e':
    partition_cls = DirichletPartition_e
elif args.data_partition == 'dirichlet_f':
    partition_cls = DirichletPartition_f
else:
    assert False, 'unknown data-partition'
# ===========================================
    

# -------------------------------------------
# define aggregation
# -------------------------------------------
_agg = (args.aggregation or '').lower()
if _agg == 'mean':
    aggregation = C_mean(honest_nodes, byzantine_nodes)
elif _agg == 'trimmed-mean':
    aggregation = C_trimmed_mean(honest_nodes, byzantine_nodes)
elif _agg == 'faba':
    aggregation = C_faba(honest_nodes, byzantine_nodes)
elif _agg == 'cc':
    # Paper 2012.10333 / byzantine-robust-optimizer: tau=100, n_iter=1
    threshold = 100
    aggregation = C_centered_clipping(honest_nodes, byzantine_nodes, threshold=threshold)
elif _agg == 'lfighter':
    aggregation = C_LFighter(honest_nodes, byzantine_nodes)
elif _agg == 'hrac':
    # HRAC: History-Residual Adaptive Clipping
    # Only suppresses per-round attack impact, no long-term weight reduction
    import os
    if partition_cls is not None:
        partition_name_str = partition_cls.__name__
    else:
        partition_name_str = 'LabelSeperation'
    
    if args.attack == 'none' or attack is None:
        log_filename = 'hrac-log-baseline.txt'
        invariant_log_filename = 'hrac-invariants-baseline.txt'
    else:
        log_filename = f'hrac-log-{attack_name}.txt'
        invariant_log_filename = f'hrac-invariants-{attack_name}.txt'
    
    log_file_path = os.path.join('record', task.name, f'Centralized_n={node_size}_b={byzantine_size}', partition_name_str, log_filename)
    invariant_log_file_path = os.path.join('record', task.name, f'Centralized_n={node_size}_b={byzantine_size}', partition_name_str, invariant_log_filename)
    print(f'[HRAC] Log file will be saved to: {log_file_path}')
    print(f'[HRAC] Invariant log file will be saved to: {invariant_log_file_path}')
    
    aggregation = C_HRAC(
        honest_nodes=honest_nodes,
        byzantine_nodes=byzantine_nodes,
        rho_b=0.98,
        rho_mu=0.95,
        rho_nu=0.87,  # ν momentum: 0.87 → ν tracks d faster (rise/decay)
        c=2.5,
        c_g=3.0,  # Global median norm cap multiplier (MSA defense)
        enable_post_residual_b_cap=False,  # 消融：Δ̃=b+r̄ 后不再乘 min(1,B/‖Δ̃‖)（仍保留入口全局帽 messages_clipped）
        enable_logging=True,
        log_interval=100,
        eps=1e-12,
        log_file=log_file_path,
        enable_invariant_checks=True,
        invariant_check_mode="log_and_raise",
        invariant_check_tol=1e-6,
        invariant_log_file=invariant_log_file_path,
        invariant_log_interval=1,
        verbose_nu_log_interval=100,        # 每 N 个 iter 打印一次 d/nu 详情（100=与主 log 对齐；1=每 iter 很刷屏；0=关闭）
    )
elif _agg == 'hsm':
    # HSM-FedAvg: History-Soft Momentum FedAvg
    # Enhanced hyperparameters for robust defense against all attack types
    # HSM-FedAvg: Clean version with simplified, well-justified hyperparameter system
    # 
    # Main Config (recommended for paper):
    #   - rho=0.92, gamma_min=0.15, lam=0.35, kappa=6.0, omega_min=0.10, tau_quantile=0.6
    #
    # HisMSA-strong Config (for ablation/appendix):
    #   - gamma_min=0.10, kappa=7.0, omega_min=0.05, tau_quantile=0.65
    #
    # Core 5 parameters (main tuning knobs):
    #   - rho: History EMA decay (0.92 recommended for HisMSA)
    #   - gamma_min: Minimum gamma (0.15 balanced, 0.10 for HisMSA-strong)
    #   - lam: Scale deviation penalty (0.35 recommended)
    #   - kappa: Sigmoid slope (6.0 balanced, 7.0 for HisMSA-strong)
    #   - omega_min: Minimum weight floor (0.10 balanced, 0.05 for HisMSA-strong)
    # Generate log file path based on task and configuration
    # Format: record/{task_name}/Centralized_n={node_size}_b={byzantine_size}/{partition_name}/hsm-log.txt
    # This matches the structure used by dump_file_in_cache
    import os
    # Get partition name from partition_cls class name
    if partition_cls is not None:
        partition_name_str = partition_cls.__name__
    else:
        partition_name_str = 'LabelSeperation'  # Default to LabelSeperation for noniid
    
    # Generate log file name with attack type
    if args.attack == 'none' or attack is None:
        log_filename = 'hsm-log-baseline.txt'
    else:
        # Use attack name in filename (e.g., hsm-log-msa.txt, hsm-log-label_flipping.txt)
        log_filename = f'hsm-log-{attack_name}.txt'
    
    log_file_path = os.path.join('record', task.name, f'Centralized_n={node_size}_b={byzantine_size}', partition_name_str, log_filename)
    # Print log file path for debugging
    print(f'[HSM-FedAvg] Log file will be saved to: {log_file_path}')
    
    aggregation = C_HSM_FedAvg(
        honest_nodes=honest_nodes,
        byzantine_nodes=byzantine_nodes,
        rho=0.9,
        beta=0.9,
        k=2.5,
        use_residual_consistency=True,
        use_cluster_ref=True,
        enable_logging=True,
        log_interval=100,
        eps=1e-12,
        log_file=log_file_path,
    )
else:
    assert False, f'unknown aggregation: {args.aggregation!r} (expected: mean, trimmed-mean, faba, cc, lfighter, hrac, hsm)'

# ===========================================

workspace = []
mark_on_title = ''
fix_seed = not args.no_fixed_seed
seed = args.seed
record_in_file = not args.without_record
step_agg = args.step_agg
# alpha = task.super_params['alpha']

# initilize optimizer
if attack_name in ['HisMSA', 'MSA'] or (attack is not None and hasattr(attack, 'name') and attack.name in ['HisMSA', 'MSA']):
    # Use CMomentum_under_HisMSA for HisMSA or MSA attack (both use model shuffling)
    env = CMomentum_under_HisMSA(aggregation=aggregation, honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes, attack=attack, step_agg = step_agg,
          weight_decay=task.weight_decay, data_package=task.data_package,
          model=task.model, loss_fn=task.loss_fn, test_fn=task.test_fn,
          initialize_fn=task.initialize_fn,
          get_train_iter=task.get_train_iter,
          get_test_iter=task.get_test_iter,
          partition_cls=partition_cls, lr_ctrl=lr_ctrl,
          fix_seed=fix_seed, seed=seed,
          **task.super_params)
elif attack_name == 'baseline' or attack is None:
    # No attack case - use CMomentum_under_DPA with attack=None
    env = CMomentum_under_DPA(aggregation=aggregation, honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes, attack=None, step_agg = step_agg,
          weight_decay=task.weight_decay, data_package=task.data_package,
          model=task.model, loss_fn=task.loss_fn, test_fn=task.test_fn,
          initialize_fn=task.initialize_fn,
          get_train_iter=task.get_train_iter,
          get_test_iter=task.get_test_iter,
          partition_cls=partition_cls, lr_ctrl=lr_ctrl,
          fix_seed=fix_seed, seed=seed,
          **task.super_params)
elif 'label' in attack_name:
    if (args.aggregation or '').lower() == 'lfighter':
        env = CMomentum_with_LFighter_under_DPA(aggregation=aggregation, honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes, attack=attack, step_agg = step_agg,
               weight_decay=task.weight_decay, data_package=task.data_package,
               model=task.model, loss_fn=task.loss_fn, test_fn=task.test_fn,
               initialize_fn=task.initialize_fn,
               get_train_iter=task.get_train_iter,
               get_test_iter=task.get_test_iter,
               partition_cls=partition_cls, lr_ctrl=lr_ctrl,
               fix_seed=fix_seed, seed=seed,
               **task.super_params)
    else:
        env = CMomentum_under_DPA(aggregation=aggregation, honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes, attack=attack, step_agg = step_agg,
              weight_decay=task.weight_decay, data_package=task.data_package,
              model=task.model, loss_fn=task.loss_fn, test_fn=task.test_fn,
              initialize_fn=task.initialize_fn,
              get_train_iter=task.get_train_iter,
              get_test_iter=task.get_test_iter,
              partition_cls=partition_cls, lr_ctrl=lr_ctrl,
              fix_seed=fix_seed, seed=seed,
              **task.super_params)
else:
    # Gradient-only attacks (bf, bit_flipping, ipm, alie from byzantine-robust-optimizer)
    env = CMomentum_under_DPA(aggregation=aggregation, honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes, attack=attack, step_agg = step_agg,
          weight_decay=task.weight_decay, data_package=task.data_package,
          model=task.model, loss_fn=task.loss_fn, test_fn=task.test_fn,
          initialize_fn=task.initialize_fn,
          get_train_iter=task.get_train_iter,
          get_test_iter=task.get_test_iter,
          partition_cls=partition_cls, lr_ctrl=lr_ctrl,
          fix_seed=fix_seed, seed=seed,
          **task.super_params)


title = '{}_{}_{}'.format(env.name, attack_name, aggregation.name)

if lr_ctrl != None:
    title = title + '_' + lr_ctrl.name
if mark_on_title != '':
    title = title + '_' + mark_on_title

data_package = task.data_package
super_params = task.super_params

# print the running information
print('=========================================================')
print('[Task] ' + task.name + ': ' + title)
print('=========================================================')
print('[Setting]')
print('{:12s} model={}'.format('[task]', task.model_name))
print('{:12s} dataset={} partition={}'.format(
    '[dataset]', data_package.name, env.partition_name))
print('{:12s} name={} aggregation={} attack={}'.format(
    '[Algorithm]', env.name, aggregation.name, attack_name))
print('{:12s} alpha={} lr={} lr_ctrl={}, weight_decay={}'.format(
    '[Optimizer]', super_params['alpha'], super_params['lr'], env.lr_ctrl.name, task.weight_decay))
print('{:12s} honest_size={}, byzantine_size={}'.format(
    '[Graph]', node_size - byzantine_size, byzantine_size))
print('{:12s} rounds={}, display_interval={}, total iterations={}'.format(
    '[Running]', env.rounds, env.display_interval, env.total_iterations))
print('{:12s} seed={}, fix_seed={}'.format('[Randomness]', seed, fix_seed))
print('{:12s} record_in_file={}'.format('[System]', record_in_file))
print('-------------------------------------------')

log('[Start Running]')
_, loss_path, acc_path = env.run()

# Extract detection history from aggregation if available (for HSM methods)
detection_recall_path = None
detection_precision_path = None
log_interval = 100  # Default log interval
if hasattr(aggregation, 'get_detection_history'):
    recall_history, precision_history = aggregation.get_detection_history()
    if recall_history:
        # Convert to simple lists: [recall_value, ...] and [precision_value, ...]
        # Iterations are implicit (same as log_interval spacing)
        detection_recall_path = [recall for _, recall in recall_history]
        detection_precision_path = [precision for _, precision in precision_history]
        # Get log_interval from aggregation if available
        if hasattr(aggregation, 'log_interval'):
            log_interval = aggregation.log_interval

record = {
    'dataset': data_package.name,
    'dataset_size': len(data_package.train_set),
    'dataset_feature_dimension': data_package.feature_dimension,
    'lr': super_params['lr'],
    'weight_decay': task.weight_decay,
    'honest_size': node_size - byzantine_size,
    'byzantine_size': byzantine_size,
    'rounds': env.rounds,
    'display_interval': env.display_interval,
    'total_iterations': env.total_iterations,
    'loss_path': loss_path,
    'acc_path': acc_path,
    'fix_seed': fix_seed,
    'seed': seed
}

# Add detection history if available
if detection_recall_path is not None:
    record['detection_recall_path'] = detection_recall_path
    record['detection_precision_path'] = detection_precision_path
    record['log_interval'] = log_interval  # Store log_interval for plotting

if record_in_file:
    path_list = [task.name, f'Centralized_n={node_size}_b={byzantine_size}', env.partition_name] + workspace
    dump_file_in_cache(title, record, path_list=path_list)
    
    # Automatically plot detection recall curve if detection history is available
    if detection_recall_path is not None and len(detection_recall_path) > 0:
        try:
            import subprocess
            import sys
            import os
            # Get the project root directory
            project_root = os.path.dirname(os.path.abspath(__file__))
            draw_script = os.path.join(project_root, 'draw_fig', 'draw_detection_recall.py')
            if os.path.exists(draw_script):
                # Pass the record file path to the drawing script
                record_file = os.path.join(project_root, 'record', *path_list, title)
                # Also pass path_list as comma-separated string for fallback loading
                path_list_str = ','.join(path_list)
                subprocess.run([sys.executable, draw_script, record_file, title, path_list_str], 
                             cwd=project_root, check=False)
                print(f'[Auto-Plot] Detection recall curve plotted automatically')
            else:
                print(f'[Auto-Plot] Warning: Drawing script not found at {draw_script}')
        except Exception as e:
            print(f'[Auto-Plot] Warning: Failed to plot detection recall curve: {e}')

print('-------------------------------------------')
