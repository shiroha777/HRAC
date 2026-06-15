from argsParser import args

from ByrdLab import FEATURE_TYPE
from ByrdLab.aggregation import C_mean, C_trimmed_mean, C_faba, C_centered_clipping, C_LFighter, C_HRAC
from ByrdLab.attack import C_gaussian, C_same_value, C_sign_flipping, C_bit_flipping, C_IPM, C_ALIE, C_MinMaxFull, C_MinMaxUnknown, C_Mimic, C_PoisonedFL, C_TauBoundary, feature_label_random, \
                            label_flipping, label_random, furthest_label_flipping, adversarial_label_flipping, feature_label_random, HisMSA, MSA
from ByrdLab.centraliedAlgorithm import CSGD, CSGD_under_DPA, CMomentum_under_DPA, CMomentum_with_LFighter_under_DPA, CMomentum_under_HisMSA
from ByrdLab.library.cache_io import dump_file_in_cache, load_file_in_cache
from ByrdLab.library.dataset import ijcnn, mnist, fashionmnist, cifar10, mnist_sorted_by_labels
from ByrdLab.library.learnRateController import ladder_lr, one_over_sqrt_k_lr
from ByrdLab.library.partition import (LabelSeperation, TrivalPartition,
                                   iidPartition, DirichletIiiPartition, DirichletMildPartition, DirichletNoniidPartition,
                                    DirichletPartition_a, DirichletPartition_b, DirichletPartition_c, DirichletPartition_d, DirichletPartition_e, DirichletPartition_f)
from ByrdLab.library.tool import log
from ByrdLab.tasks.logisticRegression import LogisticRegressionTask
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask
from ByrdLab.tasks.leastSquare import LeastSquareToySet, LeastSquareToyTask
from ByrdLab.tasks.neuralNetwork import NeuralNetworkTask

node_size = 10
DEFAULT_BYZANTINE_SIZE = 3
byzantine_size = args.byzantine_size if args.byzantine_size is not None else DEFAULT_BYZANTINE_SIZE
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
elif args.attack in ['min_max', 'minmax', 'min-max', 'min_max_full', 'minmax_full', 'min-max-full']:
    # Min-Max with known benign gradients: uses current-round honest messages.
    attack = C_MinMaxFull(honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)
elif args.attack in ['min_max_unknown', 'minmax_unknown', 'min-max-unknown']:
    # Min-Max with unknown benign gradients: uses only Byzantine messages as the reference cloud.
    attack = C_MinMaxUnknown(honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)
elif args.attack == 'mimic':
    # Mimic: all Byzantine workers copy one honest worker's current message.
    attack = C_Mimic(honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes, epsilon=0)
elif args.attack in ['poisonedfl', 'poisoned_fl']:
    # PoisonedFL-style multi-round consistency attack. sf=8 follows the source repo examples.
    attack = C_PoisonedFL(honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes, scaling_factor=8.0)
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

# Use the environment's constant learning-rate controller.
lr_ctrl = None

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
    
    hrac_ablation = args.hrac_ablation if args.hrac_ablation_experiment else None
    hrac_flags = {
        'enable_global_cap': True,
        'enable_per_client_residual_clip': True,
        'enable_nu_weighting': True,
        'enable_post_residual_b_cap': False,
    }
    if hrac_ablation == 'no_global_cap':
        hrac_flags['enable_global_cap'] = False
    elif hrac_ablation == 'no_residual_clip':
        hrac_flags['enable_per_client_residual_clip'] = False
    elif hrac_ablation == 'no_nu_weighting':
        hrac_flags['enable_nu_weighting'] = False
    elif hrac_ablation == 'global_cap_only':
        hrac_flags['enable_per_client_residual_clip'] = False
        hrac_flags['enable_nu_weighting'] = False
    elif hrac_ablation in (None, 'full'):
        pass
    else:
        assert False, f'unknown HRAC ablation: {hrac_ablation}'

    hrac_params = {
        'rho_b': args.hrac_rho_b,
        'rho_mu': args.hrac_rho_mu,
        'rho_g': args.hrac_rho_g,
        'rho_nu': 0.87,
        'rho_nu_penalty': 0.90,
        'c': 2.5,
        'c_g': 3.0,
    }

    def _fmt_hrac_param(value):
        return f'{value:.4g}'.replace('-', 'm').replace('.', 'p')

    hrac_param_tag_parts = []
    if abs(args.hrac_rho_b - 0.98) > 1e-12:
        hrac_param_tag_parts.append(f'rhob{_fmt_hrac_param(args.hrac_rho_b)}')
    if abs(args.hrac_rho_mu - 0.95) > 1e-12:
        hrac_param_tag_parts.append(f'rhomu{_fmt_hrac_param(args.hrac_rho_mu)}')
    if abs(args.hrac_rho_g - 0.95) > 1e-12:
        hrac_param_tag_parts.append(f'rhog{_fmt_hrac_param(args.hrac_rho_g)}')
    hrac_param_tag = '_'.join(hrac_param_tag_parts)

    if args.hrac_ablation_experiment:
        aggregation_name_override = f"HRAC_ablation_{args.hrac_ablation}"
        log_prefix = f"hrac-ablation-{args.hrac_ablation}"
    else:
        aggregation_name_override = "HRAC"
        log_prefix = "hrac"
    if hrac_param_tag:
        aggregation_name_override = f'{aggregation_name_override}_{hrac_param_tag}'
        log_prefix = f'{log_prefix}-{hrac_param_tag}'

    if args.attack == 'none' or attack is None:
        log_filename = f'{log_prefix}-log-baseline.txt'
        invariant_log_filename = f'{log_prefix}-invariants-baseline.txt'
    else:
        log_filename = f'{log_prefix}-log-{attack_name}.txt'
        invariant_log_filename = f'{log_prefix}-invariants-{attack_name}.txt'
    
    log_file_path = os.path.join('record', task.name, f'Centralized_n={node_size}_b={byzantine_size}', partition_name_str, log_filename)
    invariant_log_file_path = os.path.join('record', task.name, f'Centralized_n={node_size}_b={byzantine_size}', partition_name_str, invariant_log_filename)
    print(f'[HRAC] Log file will be saved to: {log_file_path}')
    print(f'[HRAC] Invariant log file will be saved to: {invariant_log_file_path}')
    print(
        '[HRAC Params] '
        f"rho_b={hrac_params['rho_b']}, rho_mu={hrac_params['rho_mu']}, "
        f"rho_g={hrac_params['rho_g']}, rho_nu={hrac_params['rho_nu']}, "
        f"c={hrac_params['c']}, c_g={hrac_params['c_g']}"
    )
    if hrac_param_tag:
        print(f'[HRAC Sensitivity] tag={hrac_param_tag}')
    if args.hrac_ablation_experiment:
        print(f'[HRAC Ablation] variant={args.hrac_ablation}, flags={hrac_flags}')
    
    aggregation = C_HRAC(
        honest_nodes=honest_nodes,
        byzantine_nodes=byzantine_nodes,
        rho_b=hrac_params['rho_b'],
        rho_mu=hrac_params['rho_mu'],
        rho_g=hrac_params['rho_g'],
        rho_nu_penalty=hrac_params['rho_nu_penalty'],
        rho_nu=hrac_params['rho_nu'],
        c=hrac_params['c'],
        c_g=hrac_params['c_g'],
        enable_global_cap=hrac_flags['enable_global_cap'],
        enable_per_client_residual_clip=hrac_flags['enable_per_client_residual_clip'],
        enable_nu_weighting=hrac_flags['enable_nu_weighting'],
        enable_post_residual_b_cap=hrac_flags['enable_post_residual_b_cap'],
        enable_logging=True,
        log_interval=100,
        eps=1e-12,
        log_file=log_file_path,
        enable_invariant_checks=not args.hrac_ablation_experiment,
        invariant_check_mode="log_and_raise",
        invariant_check_tol=1e-6,
        invariant_log_file=invariant_log_file_path,
        invariant_log_interval=1,
        verbose_nu_log_interval=100,        # 每 N 个 iter 打印一次 d/nu 详情（100=与主 log 对齐；1=每 iter 很刷屏；0=关闭）
    )
    aggregation.name = aggregation_name_override
else:
    assert False, f'unknown aggregation: {args.aggregation!r} (expected: mean, trimmed-mean, faba, cc, lfighter, hrac)'

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

if _agg == 'hrac' and args.hrac_ablation_experiment:
    record['hrac_ablation'] = args.hrac_ablation
    record['hrac_ablation_flags'] = hrac_flags

if _agg == 'hrac':
    record['hrac_params'] = hrac_params
    if hrac_param_tag:
        record['hrac_param_tag'] = hrac_param_tag

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
