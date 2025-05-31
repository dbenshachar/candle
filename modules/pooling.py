import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAvgPool2d(nn.Module):
    """
    Global average pooling for 2D images.

    Reduces each channel to a single value by averaging over spatial dimensions.

    Example:
        >>> pool = GlobalAvgPool2d()
        >>> x = torch.randn(2, 8, 16, 16)
        >>> out = pool(x)
        # out.shape == (2, 8)

    Input shape:
        Tensor of shape (batch, channels, height, width)
    Output shape:
        Tensor of shape (batch, channels)
    """
    def forward(self, x):
        return x.mean(dim=(-2, -1))

class GlobalMaxPool2d(nn.Module):
    """
    Global max pooling for 2D images.

    Reduces each channel to a single value by taking the maximum over spatial dimensions.

    Example:
        >>> pool = GlobalMaxPool2d()
        >>> x = torch.randn(2, 8, 16, 16)
        >>> out = pool(x)
        # out.shape == (2, 8)

    Input shape:
        Tensor of shape (batch, channels, height, width)
    Output shape:
        Tensor of shape (batch, channels)
    """
    def forward(self, x):
        return x.amax(dim=(-2, -1))

class AdaptiveAvgPool1d(nn.Module):
    """
    Adaptive average pooling for 1D sequences.

    Pools input to a target output size using average pooling.

    Example:
        >>> pool = AdaptiveAvgPool1d(output_size=4)
        >>> x = torch.randn(2, 8, 16)
        >>> out = pool(x)
        # out.shape == (2, 8, 4)

    Input shape:
        Tensor of shape (batch, channels, length)
    Output shape:
        Tensor of shape (batch, channels, output_size)
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return F.adaptive_avg_pool1d(x, self.output_size)

class AdaptiveAvgPool2d(nn.Module):
    """
    Adaptive average pooling for 2D images.

    Pools input to a target output size using average pooling.

    Example:
        >>> pool = AdaptiveAvgPool2d(output_size=(4, 4))
        >>> x = torch.randn(2, 8, 16, 16)
        >>> out = pool(x)
        # out.shape == (2, 8, 4, 4)

    Input shape:
        Tensor of shape (batch, channels, height, width)
    Output shape:
        Tensor of shape (batch, channels, output_height, output_width)
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, self.output_size)
