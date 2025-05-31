import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = None

    def forward(self, tensor):
        if self.beta is None:
            self.beta = torch.zeros_like(self.gamma, device=tensor.device)
        return F.layer_norm(tensor, tensor.shape[-1:], self.gamma, self.beta)
    
class PreNorm(nn.Module):
    def __init__(self, dim, function):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.function = function

    def forward(self, tensor, *args, **kwargs):
        return self.function(self.norm(tensor), *args, **kwargs)
    
class GroupNorm(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels, affine=True)

    def forward(self, tensor):
        return self.norm(tensor)
    
class ChanLayerNorm(nn.Module):
    def __init__(self, channels, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, tensor):
        var = torch.var(tensor, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(tensor, dim=1, keepdim=True)
        return (tensor - mean) * (var + self.eps).rsqrt() * self.gamma