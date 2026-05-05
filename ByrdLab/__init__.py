import argparse
import torch

try:
    from argsParser import args
except Exception:
    # Allow ByrdLab to be imported from other repos (e.g. byzantine-robust-optimizer)
    args = argparse.Namespace(gpu=0, cpu=False)

# torch.ShortTensor  ---  torch.int16
# torch.IntTensor    ---  torch.int32 / torch.int
# torch.LongTensor   ---  torch.int64

FEATURE_TYPE = torch.float64
TARGET_TYPE = torch.int16
VALUE_TYPE = torch.float64

# Use CPU when --cpu is set to lower GPU utilization
DEVICE = torch.device("cpu" if getattr(args, "cpu", False) else ("cuda:" + str(getattr(args, "gpu", 0))))