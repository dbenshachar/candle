import torch
from torch import nn
from torch.nn import functional as F

from convolution import ScaleShift
import einops

from attention import (
    ImageSelfAttention,
    ImageLinearSelfAttention,
    SequenceSelfAttention,
    SequenceImageCrossAttention,
    ImageSequenceCrossAttention,
)

class SelfAttentionLayer(nn.Module):
    """
    General self-attention layer for images or sequences.
    Args:
        mode: 'image' or 'sequence'
        use_linear: if True and mode='image', use linear attention (default True)
        **kwargs: passed to the underlying attention module
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
    Args:
        image_attn_kwargs: kwargs for image self-attention
        cross_attn_kwargs: kwargs for text-image cross-attention
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
    Args:
        seq_attn_kwargs: kwargs for sequence self-attention
        cross_attn_kwargs: kwargs for image-sequence cross-attention
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
    Simple convolutional block with normalization and activation.
    Args:
        in_channels, out_channels, kernel_size, norm_layer, activation
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
    Args:
        in_channels, out_channels, groups
    """
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.block = ScaleShift(in_channels, out_channels, groups=groups)

    def forward(self, x, scale_shift=None):
        return self.block(x, scale_shift=scale_shift)

class ScaleShiftResnetBlock(nn.Module):
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