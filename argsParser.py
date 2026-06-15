import argparse

parser = argparse.ArgumentParser(description='Mean Aggregator Is More Robust Than Robust Aggregators Under Label Poisoning Attacks on Distributed Heterogeneous Data')
    
# Arguments
parser.add_argument('--aggregation', type=str, default='mean')
parser.add_argument('--attack', type=str, default='none')
parser.add_argument('--data-partition', type=str, default='iid')
parser.add_argument('--lr-ctrl', type=str, default='1/sqrt k')

parser.add_argument('--no-fixed-seed', action='store_true',
                    help="If specifed, the random seed won't be fixed")
parser.add_argument('--seed', type=int, default=100)

parser.add_argument('--without-record', action='store_true',
                    help='If specifed, no file of running record and log will be left')
parser.add_argument('--step-agg', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--cpu', action='store_true',
                    help='Use CPU instead of GPU to lower GPU utilization')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Training batch size (smaller = less GPU memory and load, default 32)')
parser.add_argument('--prob', type=float, default=1.0)
parser.add_argument('--byzantine-size', type=int, default=None,
                    help='Override the number of Byzantine clients. If omitted, use the script default.')
parser.add_argument('--hrac-ablation-experiment', action='store_true',
                    help='Run HRAC under an ablation name so records do not overwrite normal HRAC.')
parser.add_argument('--hrac-ablation', type=str, default='full',
                    choices=['full', 'no_global_cap', 'no_residual_clip', 'no_nu_weighting', 'global_cap_only'],
                    help='HRAC ablation variant to use when --aggregation hrac.')
parser.add_argument('--hrac-rho-b', type=float, default=0.98,
                    help='HRAC EMA decay for the per-client history centre.')
parser.add_argument('--hrac-rho-mu', type=float, default=0.95,
                    help='HRAC EMA decay for the residual scale statistic.')
parser.add_argument('--hrac-rho-g', type=float, default=0.95,
                    help='HRAC EMA decay for the global median-norm scale.')

# parse_known_args() so that when ByrdLab is imported from other scripts (e.g. byzantine-robust-optimizer),
# unknown flags like --use-cuda, --agg, --momentum do not cause errors
args, _ = parser.parse_known_args()
gpu = args.gpu
