import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import numpy as np
from typing import List, Tuple, Any, Optional, Union, Set

def zip_tensors(tup_list):
    arr0, arr1, arr2 = zip(*tup_list)
    if type(arr2[0]) is int:
        arr0 = torch.stack(arr0, dim=0)
        arr1 = torch.tensor(arr1, dtype=torch.long)
        arr2 = torch.tensor(arr2, dtype=torch.long)
    else:
        arr0 = torch.cat(arr0, dim=0)
        arr1 = [x for a in arr1 for x in a]
        arr1 = torch.tensor(arr1, dtype=torch.long)
        arr2 = torch.cat(arr2, dim=0)
    return arr0, arr1, arr2

def create_pad_tensor(alist):
    max_len = max([len(a) for a in alist])
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.tensor(alist, dtype=torch.long)

def index_scatter(sub_data, all_data, index):
    d0, d1 = all_data.size()
    buf = torch.zeros_like(all_data).scatter_(0, index.repeat(d1, 1).t(), sub_data)
    mask = torch.ones(d0, device=all_data.device).scatter_(0, index, 0)
    return all_data * mask.unsqueeze(-1) + buf


def stack_pad_tensor(tensor_list):
    max_len = max([t.size(0) for t in tensor_list])
    for i,tensor in enumerate(tensor_list):
        pad_len = max_len - tensor.size(0)
        tensor_list[i] = F.pad( tensor, (0,0,0,pad_len) )
    return torch.stack(tensor_list, dim=0)


def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)


class EncOptimizer(Optimizer):

    def __init__(self, optimizer: Optimizer, enc_opt: Optional[Optimizer]) -> None:
        enc_params = []
        if enc_opt is not None:
            enc_params = enc_opt.param_groups
        super().__init__(optimizer.param_groups + enc_params, {})
        self.optimizer = optimizer
        self.enc_opt = enc_opt

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()
        if self.enc_opt is not None:
            self.enc_opt.zero_grad()

    def step(self, closure: Optional[Any] = None) -> None:
        self.optimizer.step(closure)
        if self.enc_opt is not None:
            self.enc_opt.step(closure)


def build_mlp(in_dim: int,
              h_dim: Union[int, List],
              out_dim: int = None,
              dropout_p: float = 0.2) -> nn.Sequential:
    """Builds an MLP.
    Parameters
    ----------
    in_dim: int,
        Input dimension of the MLP
    h_dim: int,
        Hidden layer dimension of the MLP
    out_dim: int, default None
        Output size of the MLP. If None, a Linear layer is returned, with ReLU
    dropout_p: float, default 0.2,
        Dropout probability
    """
    if isinstance(h_dim, int):
        h_dim = [h_dim]

    sizes = [in_dim] + h_dim
    mlp_size_tuple = list(zip(*(sizes[:-1], sizes[1:])))

    if isinstance(dropout_p, float):
        dropout_p = [dropout_p] * len(mlp_size_tuple)

    layers = []

    for idx, (prev_size, next_size) in enumerate(mlp_size_tuple):
        layers.append(nn.Linear(prev_size, next_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_p[idx]))

    if out_dim is not None:
        layers.append(nn.Linear(sizes[-1], out_dim))

    return nn.Sequential(*layers)
