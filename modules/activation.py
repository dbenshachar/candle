import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU activation function module.

    Splits the input tensor in half along the last dimension, applies SiLU (Swish) to the gate half, and multiplies elementwise with the other half.

    Example:
        >>> act = SwiGLU()
        >>> x = torch.randn(2, 8)
        >>> out = act(x)
        # x.shape == (2, 8), out.shape == (2, 4)

    Input shape:
        Tensor of shape (..., 2 * D)
    Output shape:
        Tensor of shape (..., D)
    """
    def forward(self, tensor):
        tensor, gate = tensor.chunk(2, dim=-1)
        return F.silu(gate) * tensor
    
class GEGLU(nn.Module):
    """
    GEGLU activation function module.

    Splits the input tensor in half along the last dimension, applies GELU to the gate half, and multiplies elementwise with the other half.

    Example:
        >>> act = GEGLU()
        >>> x = torch.randn(2, 8)
        >>> out = act(x)
        # x.shape == (2, 8), out.shape == (2, 4)

    Input shape:
        Tensor of shape (..., 2 * D)
    Output shape:
        Tensor of shape (..., D)
    """
    def forward(self, tensor):
        tensor, gate = tensor.chunk(2, dim=-1)
        return tensor * F.gelu(gate)