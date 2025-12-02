from math import isnan
import sys
import torch

class NanOrInfError(Exception):
    sys.exit(1)

def check_nan_inf(tensor, out_str='Tensor contains Nan or Inf!'):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise NanOrInfError(out_str)