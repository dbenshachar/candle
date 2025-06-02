import torch
from torch import nn
from torch.nn import functional as F

from modules.convolution import ScaleShift, ChannelAttention
from modules.norm import ChanLayerNorm
import einops
import einops.layers.torch as layers

from modules.attention import (
    ImageSelfAttention,
    ImageLinearSelfAttention,
    SequenceSelfAttention,
    SequenceImageCrossAttention,
    ImageSequenceCrossAttention,
)

class SelfAttentionLayer(nn.Module):
    """
    General self-attention layer for images or sequences.

    Example:
        >>> layer = SelfAttentionLayer(mode='image', use_linear=True, channels=32, heads=4)
        >>> x = torch.randn(2, 32, 16, 16)
        >>> out = layer(x)
        # out.shape == (2, 32, 16, 16)

        >>> layer = SelfAttentionLayer(mode='sequence', dim=64, heads=4)
        >>> x = torch.randn(2, 10, 64)
        >>> out = layer(x)
        # out.shape == (2, 10, 64)

    Args:
        mode (str): 'image' or 'sequence'.
        use_linear (bool): If True and mode='image', use linear attention (default True).
        **kwargs: Additional arguments passed to the underlying attention module.

    Input shape:
        For mode='image': (batch, channels, height, width)
        For mode='sequence': (batch, seq_len, dim)
    Output shape:
        Same as input shape.
    """
    def __init__(self, mode='image', use_linear=True, **kwargs):
        super().__init__()
        assert mode in ('image', 'sequence')
        if mode == 'image':
            if use_linear:
                self.attn = ImageLinearSelfAttention(**kwargs)
            else:
                self.attn = ImageSelfAttention(**kwargs)
        else:
            self.attn = SequenceSelfAttention(**kwargs)

    def forward(self, x, mask=None):
        if isinstance(self.attn, SequenceSelfAttention):
            return self.attn(x, mask=mask)
        return self.attn(x)

class ImageTextAttentionLayer(nn.Module):
    """
    Layer combining image self-attention and text-image cross-attention.

    Example:
        >>> image_attn_kwargs = dict(channels=32, heads=4)
        >>> cross_attn_kwargs = dict(dim=64, heads=2)
        >>> layer = ImageTextAttentionLayer(image_attn_kwargs, cross_attn_kwargs)
        >>> image = torch.randn(2, 32, 16, 16)
        >>> text = torch.randn(2, 10, 64)
        >>> image_out, text_out = layer(image, text)
        # image_out.shape == (2, 32, 16, 16)
        # text_out.shape == (2, 10, 64)

    Args:
        image_attn_kwargs (dict): Keyword arguments for image self-attention.
        cross_attn_kwargs (dict): Keyword arguments for text-image cross-attention.

    Input shape:
        image: (batch, channels, height, width)
        text: (batch, seq_len, dim)
    Output shape:
        image_out: (batch, channels, height, width)
        text_out: (batch, seq_len, dim)
    """
    def __init__(self, image_attn_kwargs, cross_attn_kwargs):
        super().__init__()
        self.image_attn = ImageSelfAttention(**image_attn_kwargs)
        self.cross_attn = SequenceImageCrossAttention(**cross_attn_kwargs)

    def forward(self, image, text, mask=None):
        image_out = self.image_attn(image)
        text_out = self.cross_attn(text, image_out, mask=mask)
        return image_out, text_out

class SequenceImageAttentionLayer(nn.Module):
    """
    Layer combining sequence self-attention and image-sequence cross-attention.

    Use case:
        This layer is designed for multimodal models that process both sequential (e.g., text) and image data. It applies self-attention to a sequence and then cross-attends the image features to the sequence features, enabling information flow between modalities.

    Args:
        seq_attn_kwargs (dict): Arguments for sequence self-attention.
        cross_attn_kwargs (dict): Arguments for image-sequence cross-attention.

    Input shape:
        seq: (batch, seq_len, dim) - sequence input (e.g., text embeddings)
        image: (batch, image_len, dim) - image input (e.g., flattened image patches or tokens)
        mask (optional): (batch, seq_len) or (batch, image_len) - attention mask
    Output shape:
        seq_out: (batch, seq_len, dim) - attended sequence output
        image_out: (batch, image_len, dim) - attended image output
    """
    def __init__(self, seq_attn_kwargs, cross_attn_kwargs):
        super().__init__()
        self.seq_attn = SequenceSelfAttention(**seq_attn_kwargs)
        self.cross_attn = ImageSequenceCrossAttention(**cross_attn_kwargs)

    def forward(self, seq, image, mask=None):
        seq_out = self.seq_attn(seq, mask=mask)
        image_out = self.cross_attn(image, seq_out, mask=mask)
        return seq_out, image_out

class ConvLayer(nn.Module):
    """
    General convolutional layer with normalization and activation.

    Use case:
        This layer is a flexible building block for convolutional neural networks, supporting configurable normalization and activation. Useful for image feature extraction or as a component in larger architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel (default: 3).
        norm_layer (callable): Normalization layer class (default: nn.GroupNorm).
        groups (int): Number of groups for normalization (default: 8).
        activation (callable): Activation function class (default: nn.SiLU).

    Input shape:
        x: (batch, in_channels, height, width)
    Output shape:
        (batch, out_channels, height, width)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, norm_layer=nn.GroupNorm, groups=8, activation=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm = norm_layer(groups, out_channels)
        self.act = activation()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DiffusionConvLayer(nn.Module):
    """
    Convolutional block for diffusion models using scale-shift.

    Example:
        >>> layer = DiffusionConvLayer(in_channels=32, out_channels=64)
        >>> x = torch.randn(2, 32, 16, 16)
        >>> out = layer(x)
        # out.shape == (2, 64, 16, 16)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        groups (int): Number of groups for normalization.

    Input shape:
        (batch, in_channels, height, width)
    Output shape:
        (batch, out_channels, height, width)
    """
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.block = ScaleShift(in_channels, out_channels, groups=groups)

    def forward(self, x, scale_shift=None):
        return self.block(x, scale_shift=scale_shift)

class ScaleShiftResnetBlock(nn.Module):
    """
    Residual block with scale-shift normalization and optional time embedding.

    Example:
        >>> block = ScaleShiftResnetBlock(dim=32, dim_out=64, time_emb_dim=128)
        >>> x = torch.randn(2, 32, 16, 16)
        >>> t = torch.randn(2, 128)
        >>> out = block(x, time_emb=t)
        # out.shape == (2, 64, 16, 16)

    Args:
        dim (int): Input dimension.
        dim_out (int): Output dimension.
        time_emb_dim (int, optional): Dimension of time embedding.
        groups (int): Number of groups for normalization.

    Input shape:
        x: (batch, dim, height, width)
        time_emb (optional): (batch, time_emb_dim)
    Output shape:
        (batch, dim_out, height, width)
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )

        self.block1 = ScaleShift(dim, dim_out, groups=groups)
        self.block2 = ScaleShift(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = einops.rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
    
class SqueezeAndExciteLayer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.channel_attention = ChannelAttention(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class UNetBlock(nn.Module):
    """
    Residual block with ScaleShift and optional time embedding.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim=None, groups=8):
        super().__init__()
        self.block1 = ScaleShift(in_ch, out_ch, groups=groups)
        self.block2 = ScaleShift(out_ch, out_ch, groups=groups)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch * 2)
            )
        else:
            self.time_mlp = None

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.time_mlp is not None and time_emb is not None:
            t = self.time_mlp(time_emb)
            t = t[..., None, None]
            scale, shift = t.chunk(2, dim=1)
            scale_shift = (scale, shift)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    
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