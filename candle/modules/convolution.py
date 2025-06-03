import torch
from torch import nn
from torch.nn import functional as F
import einops
import einops.layers.torch as layers
    
def Conv(channels, out_channels=None, kernel_size=3):
    """
    Depthwise convolution utility function.

    Returns a depthwise 2D convolution layer with padding and stride 1.

    Example:
        >>> conv = Conv(16)
        >>> x = torch.randn(1, 16, 32, 32)
        >>> out = conv(x)
        # out.shape == (1, 16, 32, 32)

    Args:
        channels (int): Number of input channels.
        out_channels (int, optional): Number of output channels (default: same as input).
        kernel_size (int): Size of the convolution kernel (default: 3).

    Input shape:
        Tensor of shape (batch, channels, height, width)
    Output shape:
        Tensor of shape (batch, out_channels, height, width)
    """
    if out_channels is None:
        out_channels = channels
    return nn.Conv2d(channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels, stride=1)

class ScaleShift(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)