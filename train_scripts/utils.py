import warnings
import torch
import torch.nn as nn
import math
import numpy as np
from typing import Union, Iterable
from torch import inf
import random
import os
_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]
def calculate_depart_shape(x, p):
    xp = int(np.ceil(x / p)) * p
    return xp

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    keep_prob = 1 - drop_prob
    if drop_prob == 0. or not training:
        return x.div_(keep_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_."
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()

        # transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def window_partition(x, window_size):
    B, Z, H, W, C = x.shape
    x = x.view(B, Z // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 5, 1, 3, 2, 4, 6, 7).contiguous().view(-1, (Z // window_size[0]) * (H // window_size[1]), window_size[0], window_size[1], window_size[2], C)
    return windows


def cal_lat_weight():
    lat_tensor = torch.linspace(math.pi / 2, -math.pi / 2, steps=721)
    lat_step = math.pi / (721 - 1)
    half_step = lat_step / 2.0
    theta_u = torch.clamp(lat_tensor + half_step, max=math.pi/2)
    theta_l = torch.clamp(lat_tensor - half_step, min=-math.pi/2)
    lat_weight = torch.sin(theta_u) - torch.sin(theta_l)
    lat_weight = lat_weight / lat_weight.mean()
    lat_weight = lat_weight.to(torch.float32).view(-1, 1)
    return lat_weight.cuda()

def cal_lat_weight_pangu():
    lat_tensor = torch.linspace(-math.pi / 2, math.pi / 2, steps=721)
    lat_weight = torch.abs(torch.cos(lat_tensor))
    lat_weight = lat_weight.cuda()
    lat_weight = lat_weight.view(-1, 1)
    lat_weight_ratio = 721 / lat_weight.sum()
    lat_weight = lat_weight * lat_weight_ratio
    return lat_weight

def cal_mae(output, target, lat_weight):
    residual = torch.abs((output - target) * lat_weight)
    mae = torch.mean(residual)
    return mae

def cal_rmse(output, target, lat_weight):
    residual2 = (output - target) * (output - target) * lat_weight
    rmse = torch.sqrt(torch.mean(residual2))
    return rmse

def cal_mse(output, target, lat_weight):
    residual2 = (output - target) * (output - target) * lat_weight
    mse = torch.mean(residual2)
    return mse

def cal_ens(output, target, lat_weight):
    output = torch.mean(output,dim=1)
    residual2 = (output - target) * (output - target) * lat_weight
    mse = torch.mean(residual2)
    return mse

def parse_list(string1):
    list1 = [int(x) for x in string1.split('_')]
    return list1

def cal_crps(ensemble, obs, lat_weight):
    B, N, H, W = ensemble.shape
    term1 = torch.mean(torch.abs(ensemble - obs.unsqueeze(1)), dim=1)
    term2_accumulator = torch.zeros(B, H, W, device=ensemble.device, dtype=torch.float32)
    for i in range(N):
        for j in range(i+1, N):
            pairwise_diff = torch.abs(ensemble[:, i, :, :] - ensemble[:, j, :, :])
            term2_accumulator += pairwise_diff
    term2 = term2_accumulator / (N * (N-1))
    crps = (term1 - term2) * lat_weight
    return torch.mean(crps)


def cal_spread(ensemble, lat_weight):
    var = torch.var(ensemble, dim=1, unbiased=True) * lat_weight
    return torch.mean(var)


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}




def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, clip_grad = True) -> torch.Tensor:
    r"""
    Copy from torch.nn.utils.clip_grad_norm_

    Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)

    if clip_grad:
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
        # gradient_cliped = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    return total_norm