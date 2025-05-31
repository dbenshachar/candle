import torch
import torch.nn as nn
import math
import einops

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for sequences.

    Adds fixed sinusoidal positional encodings to the input tensor.

    Example:
        >>> emb = SinusoidalPositionalEmbedding(d_model=16)
        >>> x = torch.randn(2, 10, 16)
        >>> out = emb(x)
        # out.shape == (2, 10, 16)

    Args:
        d_model (int): Embedding dimension.
        max_seq_len (int): Maximum sequence length (default: 5000).
        dropout (float): Dropout probability (default: 0.0).

    Input shape:
        Tensor of shape (batch, seq_len, d_model)
    Output shape:
        Tensor of shape (batch, seq_len, d_model)
    """
    def __init__(self, d_model, max_seq_len=5000, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        seq_len = x.shape[1]
        pe_sliced = self.pe[:, :seq_len, :].to(x.device, dtype=x.dtype)
        x = x + pe_sliced
        return self.dropout(x)
    
class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding for sequences.

    Applies rotary position encodings to the input tensor.

    Example:
        >>> emb = RotaryEmbedding(dim=8)
        >>> x = torch.randn(2, 10, 8)
        >>> out = emb(x)
        # out.shape == (2, 10, 8)

    Args:
        dim (int): Embedding dimension (must be even).
        max_seq_len (int): Maximum sequence length (default: 2048).
        base (float): Base for frequency calculation (default: 10000).

    Input shape:
        Tensor of shape (batch, seq_len, dim) or (batch, heads, seq_len, dim)
    Output shape:
        Tensor of same shape as input
    """
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding dimension `dim` must be even, but got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=max_seq_len, device=torch.device("cpu"), dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, offset=0):
        seq_len = x.shape[1]
        if seq_len + offset > self.max_seq_len or self.cos_cached.device != x.device or self.cos_cached.dtype != x.dtype:
            self._set_cos_sin_cache(seq_len + offset, x.device, x.dtype)
            self.max_seq_len = seq_len + offset

        cos = self.cos_cached[:, offset:seq_len + offset, :, :].squeeze(2)
        sin = self.sin_cached[:, offset:seq_len + offset, :, :].squeeze(2)

        if x.ndim == 3:
            cos = cos.squeeze(2)
            sin = sin.squeeze(2)
        
        x_rotated = (x * cos) + (self._rotate_half(x) * sin)
        return x_rotated
    
class SinusoidalPositionalEmbeddingVision(nn.Module):
    """
    Sinusoidal positional embedding for 2D images.

    Adds 2D sinusoidal positional encodings to the input tensor.

    Example:
        >>> emb = SinusoidalPositionalEmbeddingVision(dim=8, height_or_width=16)
        >>> x = torch.randn(2, 8, 16, 16)
        >>> out = emb(x)
        # out.shape == (2, 24, 16, 16)

    Args:
        dim (int): Embedding dimension.
        height_or_width (int): Height or width of the image.
        theta (float): Base for frequency calculation (default: 10000).

    Input shape:
        Tensor of shape (batch, dim, height, width)
    Output shape:
        Tensor of shape (batch, dim + 2 * dim, height, width)
    """
    def __init__(self, dim, height_or_width, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

        hw_range = torch.arange(height_or_width)
        coors = torch.stack(torch.meshgrid(hw_range, hw_range, indexing = 'ij'), dim = -1)
        coors = einops.rearrange(coors, 'h w c -> h w c')
        self.register_buffer('coors', coors, persistent = False)

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        emb = einops.rearrange(self.coors, 'h w c -> h w c 1') * einops.rearrange(emb, 'j -> 1 1 1 j')
        fourier = torch.cat((emb.sin(), emb.cos()), dim = -1)
        fourier = einops.repeat(fourier, 'h w c d -> b (c d) h w', b = x.shape[0])
        return torch.cat((x, fourier), dim = 1)
    
class TimeEmbedding(nn.Module):
    """
    Standard sinusoidal time embedding for diffusion models.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb