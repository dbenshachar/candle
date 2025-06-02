import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.convolution import ScaleShift
from .modules.norm import ChanLayerNorm
import math

from .modules.embedding import TimeEmbedding
from .layers import Downsample, Upsample, UNetBlock

class UNet(nn.Module):
    """
    U-Net with ScaleShift blocks for diffusion models.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        base_channels (int): Base channel count.
        channel_mults (tuple): Multipliers for each level.
        num_res_blocks (int): Number of residual blocks per level.
        time_emb_dim (int): Time embedding dimension.
        groups (int): Groups for normalization.
        use_time (bool): Whether to use time embedding.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        groups=8,
        use_time=True,
    ):
        super().__init__()
        self.use_time = use_time
        if use_time:
            self.time_mlp = nn.Sequential(
                TimeEmbedding(base_channels),
                nn.Linear(base_channels, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.time_mlp = None

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling
        in_ch = base_channels
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        channels = [in_ch]
        for mult in channel_mults:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(UNetBlock(in_ch, out_ch, time_emb_dim, groups=groups))
                in_ch = out_ch
                channels.append(in_ch)
            self.downsamples.append(Downsample(in_ch))

        # Middle
        self.mid = nn.Sequential(
            UNetBlock(in_ch, in_ch, time_emb_dim, groups=groups),
            UNetBlock(in_ch, in_ch, time_emb_dim, groups=groups),
        )

        # Upsampling
        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(UNetBlock(in_ch + channels.pop(), out_ch, time_emb_dim, groups=groups))
                in_ch = out_ch
            self.upsamples.append(Upsample(in_ch))

        # Final normalization and conv
        self.final_norm = ChanLayerNorm(in_ch)
        self.final_conv = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, x, t=None):
        """
        Args:
            x: (batch, in_channels, height, width)
            t: (batch,) time steps or None if use_time=False
        Returns:
            (batch, out_channels, height, width)
        """
        if self.use_time:
            t_emb = self.time_mlp(t)
        else:
            t_emb = None
        h = self.init_conv(x)
        hs = [h]

        # Down path
        for down, downsample in zip(self.downs, self.downsamples):
            h = down(h, t_emb)
            hs.append(h)
            h = downsample(h)

        # Middle
        h = self.mid(h, t_emb) if isinstance(self.mid, nn.Sequential) else self.mid(h, t_emb)

        # Up path
        for up, upsample in zip(self.ups, self.upsamples):
            h = torch.cat([h, hs.pop()], dim=1)
            h = up(h, t_emb)
            h = upsample(h)

        h = self.final_norm(h)
        return self.final_conv(h)

    def generate(
        self,
        shape,
        num_steps=50,
        device=None,
        diffusion=True,
        eta=0.0,
        **kwargs
    ):
        """
        Generate an image using the UNet.
        If diffusion is enabled, uses a simple DDPM-like sampling loop.

        Args:
            shape: (batch, channels, height, width)
            num_steps: Number of diffusion steps (if enabled)
            device: torch device
            diffusion: Whether to use diffusion sampling
            eta: Noise scale for DDIM (if used)
            **kwargs: Additional args for forward()
        Returns:
            Generated image tensor
        """
        device = device or next(self.parameters()).device
        if self.use_time and diffusion:
            # DDPM-like sampling
            batch, channels, height, width = shape
            x = torch.randn(shape, device=device)
            timesteps = torch.linspace(1, 0, num_steps, device=device)
            for i, t in enumerate(timesteps):
                t_batch = torch.full((batch,), t, device=device)
                noise_pred = self(x, t_batch, **kwargs)
                # Simple Euler step (for demonstration, not production)
                if i < num_steps - 1:
                    noise = torch.randn_like(x) if eta > 0 else 0
                    x = x - noise_pred / num_steps + eta * noise / (num_steps ** 0.5)
                else:
                    x = x - noise_pred / num_steps
            return x
        else:
            # No diffusion: just run forward on random input
            x = torch.randn(shape, device=device)
            return self(x, None if self.use_time else None, **kwargs)
