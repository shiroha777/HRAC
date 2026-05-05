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

# parse_known_args() so that when ByrdLab is imported from other scripts (e.g. byzantine-robust-optimizer),
# unknown flags like --use-cuda, --agg, --momentum do not cause errors
args, _ = parser.parse_known_args()
gpu = args.gpu
