import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    Custom Layer Normalization module.

    Applies layer normalization over the last dimension of the input tensor.

    Example:
        >>> norm = LayerNorm(8)
        >>> x = torch.randn(2, 8)
        >>> out = norm(x)
        # out.shape == (2, 8)

    Args:
        dim (int): Number of features in the last dimension.

    Input shape:
        Tensor of shape (..., dim)
    Output shape:
        Tensor of shape (..., dim)
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = None

    def forward(self, tensor):
        if self.beta is None:
            self.beta = torch.zeros_like(self.gamma, device=tensor.device)
        return F.layer_norm(tensor, tensor.shape[-1:], self.gamma, self.beta)
    
class PreNorm(nn.Module):
    """
    Pre-normalization wrapper for a function/module.

    Applies LayerNorm before passing the tensor to the given function.

    Example:
        >>> fn = nn.Linear(8, 8, bias=False)
        >>> prenorm = PreNorm(8, fn)
        >>> x = torch.randn(2, 8)
        >>> out = prenorm(x)
        # out.shape == (2, 8)

    Args:
        dim (int): Number of features in the last dimension.
        function (nn.Module): Function or module to apply after normalization.

    Input shape:
        Tensor of shape (..., dim)
    Output shape:
        Tensor of shape (..., dim)
    """
    def __init__(self, dim, function):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.function = function

    def forward(self, tensor, *args, **kwargs):
        return self.function(self.norm(tensor), *args, **kwargs)
    
class GroupNorm(nn.Module):
    """
    Group Normalization module.

    Applies group normalization over the input tensor.

    Example:
        >>> norm = GroupNorm(16, groups=4)
        >>> x = torch.randn(2, 16, 8, 8)
        >>> out = norm(x)
        # out.shape == (2, 16, 8, 8)

    Args:
        channels (int): Number of channels.
        groups (int): Number of groups (default: 8).

    Input shape:
        Tensor of shape (batch, channels, ...)
    Output shape:
        Tensor of same shape as input
    """
    def __init__(self, channels, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels, affine=True)

    def forward(self, tensor):
        return self.norm(tensor)
    
class ChanLayerNorm(nn.Module):
    """
    Channel-wise Layer Normalization for images.

    Applies normalization across the channel dimension for 4D tensors.

    Example:
        >>> norm = ChanLayerNorm(8)
        >>> x = torch.randn(2, 8, 16, 16)
        >>> out = norm(x)
        # out.shape == (2, 8, 16, 16)

    Args:
        channels (int): Number of channels.
        eps (float): Epsilon for numerical stability (default: 1e-5).

    Input shape:
        Tensor of shape (batch, channels, height, width)
    Output shape:
        Tensor of same shape as input
    """
    def __init__(self, channels, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, tensor):
        var = torch.var(tensor, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(tensor, dim=1, keepdim=True)
        return (tensor - mean) * (var + self.eps).rsqrt() * self.gamma