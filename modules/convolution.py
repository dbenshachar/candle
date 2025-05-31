import torch
from torch import nn
from torch.nn import functional as F
import einops.layers.torch as layers
from norm import ChanLayerNorm

class ShiftPoolingTransformer(nn.Module):
    """
    Transformer block for images using shift pooling and patch tokenization.

    Applies spatial shifts to the input image, concatenates the shifted images, and projects to patch tokens.

    Example:
        >>> model = ShiftPoolingTransformer(dim=64, patch_size=2, channels=3)
        >>> x = torch.randn(1, 3, 32, 32)
        >>> out = model(x)
        # out.shape == (1, 64, 16, 16) for patch_size=2

    Args:
        dim (int): Output dimension after patch projection.
        patch_size (int): Size of each patch.
        channels (int): Number of input channels.
        norm (callable): Normalization layer class (default: ChanLayerNorm).

    Input shape:
        Tensor of shape (batch, channels, height, width)
    Output shape:
        Tensor of shape (batch, dim, height // patch_size, width // patch_size)
    """
    def __init__(self, dim, patch_size, channels, norm=ChanLayerNorm):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            layers.Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            norm(patch_dim),
            nn.Conv2d(patch_dim, dim, 1)
        )

    def forward(self, image):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_img = list(map(lambda shift: F.pad(image, shift), shifts))
        img_with_shifts = torch.cat((image, *shifted_img), dim = 1)
        return self.to_patch_tokens(img_with_shifts)
    
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