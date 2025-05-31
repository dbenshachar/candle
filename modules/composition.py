import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    """
    Residual wrapper for a function/module.

    Applies the given function to the input tensor and adds the input back (residual connection).

    Example:
        >>> fn = nn.Linear(10, 10, bias=False)
        >>> residual = Residual(fn)
        >>> x = torch.randn(2, 10)
        >>> out = residual(x)
        # out.shape == (2, 10)

    Args:
        function (nn.Module): A module or function to apply to the input tensor.

    Input shape:
        Tensor of shape (..., D)
    Output shape:
        Tensor of shape (..., D)
    """
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, tensor):
        return self.function(tensor) + tensor
    
class Identity(nn.Module):
    """
    Identity module that returns the input tensor unchanged.

    Example:
        >>> identity = Identity()
        >>> x = torch.randn(2, 5)
        >>> out = identity(x)
        # out.shape == (2, 5)

    Input shape:
        Tensor of any shape
    Output shape:
        Tensor of the same shape as input
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, tensor):
        return tensor
    
class FeedForward(nn.Module):
    """
    FeedForward neural network block with optional normalization and activation.

    Consists of two linear layers (biases off), an activation, normalization, and dropout.

    Example:
        >>> ff = FeedForward(dim=16, activation=nn.GELU, norm=nn.LayerNorm, scale=4, dropout=0.1)
        >>> x = torch.randn(8, 16)
        >>> out = ff(x)
        # out.shape == (8, 16)

    Args:
        dim (int): Input and output dimension.
        activation (callable): Activation function class (default: nn.ReLU).
        norm (callable or None): Normalization layer class (default: None, uses Identity).
        scale (float): Expansion factor for hidden dimension (default: 4).
        dropout (float): Dropout probability (default: 0.).

    Input shape:
        Tensor of shape (..., dim)
    Output shape:
        Tensor of shape (..., dim)
    """
    def __init__(self, dim, activation=nn.ReLU, norm=None, scale=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * scale)

        if norm is None:
          norm = Identity()

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias = False),
            activation(),
            norm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(self, tensor):
        return self.net(tensor)