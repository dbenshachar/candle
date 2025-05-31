import torch
from torch import nn
from torch.nn import functional as F

import einops

from norm import ChanLayerNorm, LayerNorm
from embedding import RotaryEmbedding
from convolution import Conv

class ImageSelfAttention(nn.Module):
    def __init__(self, channels, heads, dim_head=16, ff_mult=4, use_rope=True, max_seq_len=2048):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_rope = use_rope

        self.norm = ChanLayerNorm(channels)

        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(channels, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, channels, 1)
        self.primer_convs = nn.ModuleList([Conv(hidden_dim) for _ in range(3)])

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(dim=dim_head, max_seq_len=max_seq_len)
        else:
            self.rotary_emb = None

    def forward(self, tensor):
        height, width = tensor.shape[2], tensor.shape[3]
        tensor = self.norm(tensor)
        
        q, k, v = self.to_qkv(tensor).chunk(3, dim=1)
        q, k, v = [ds_conv(t) for ds_conv, t in zip(self.primer_convs, (q, k, v))]
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), (q, k, v))

        if self.use_rope and self.rotary_emb is not None:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        q = q * self.scale
        
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = einops.rearrange(out, 'b h (x y) d -> b (h d) x y', x=height, y=width)
        return self.to_out(out)
    
class Attend(nn.Module):
    def __init__(self, heads: int, dropout: float = 0., use_flash_attn: bool = False, is_casual: bool = False):
        super().__init__()
        self.is_casual = is_casual
        self.use_flash_attn = use_flash_attn
        self.heads = heads
        self.dropout_p = dropout
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        seq_len_q = q.shape[-2]
        seq_len_k = k.shape[-2]
        device = q.device

        if self.use_flash_attn:

            attn_mask_sdp = None
            if mask is not None:
                attn_mask_sdp = einops.rearrange(mask, 'b j -> b 1 1 j')
            with nn.attention.sdpa_kernel(nn.attention.SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask_sdp,
                    dropout_p=self.dropout_p if self.training else 0.,
                    is_causal=self.is_casual
                )
            return out

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        if mask is not None:
            expanded_mask = einops.rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~expanded_mask, -torch.finfo(sim.dtype).max)

        if self.is_casual:
            causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=device, dtype=torch.bool), diagonal=1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        return out

class SequenceSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int = 64, dropout: float = 0., is_casual: bool = False, use_flash_attn: bool = False, use_rope: bool = False, max_seq_len: int = 2048):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_rope = use_rope

        hidden_dim = dim_head * heads

        self.norm = LayerNorm(dim)
        self.attend = Attend(heads=heads, dropout=dropout, use_flash_attn=use_flash_attn, is_casual=is_casual)

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(dim=dim_head, max_seq_len=max_seq_len)
        else:
            self.rotary_emb = None

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        tensor = self.norm(tensor)
        
        q, k, v = self.to_qkv(tensor).chunk(3, dim=-1) # Split along last dimension
        
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        
        if self.use_rope and self.rotary_emb is not None:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)
        
        q = q * self.scale

        out = self.attend(q, k, v, mask=mask)
        
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
class ImageLinearSelfAttention(nn.Module):
    def __init__(self, channels, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        hidden_dim = channels * heads * dim_head
        self.to_qkv = nn.Conv2d(channels, hidden_dim * 3, kernel_size=1)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True),
            nn.GroupNorm(1, channels),
        )

    def forward(self, image):
        height, width = image.shape[2], image.shape[3]
        qkv = self.to_qkv(image).chunk(3, dim=1)
        q, k, v = map(lambda t : einops.rearrange(t, 'b (c h) x y -> b h c (x y)', h=self.heads), (qkv))

        q, k = q.softmax(dim=-2), k.softmax(dim=-1)
        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=height, y=width)
        return self.to_out(out)
    
class SequenceImageCrossAttention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=16):
        super().__init__()
        self.scale = dim ** -0.5
        self.heads = heads

        inner_dim = heads * dim_head
        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
    
    def forward(self, text, image_embeds, mask=None):
        text = self.norm(text)
        q = self.to_q(text)
        k, v = self.to_kv(image_embeds).chunk(2, dim=-1)

        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q = q * self.scale

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        if mask is not None:
            mask = einops.rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
class SequenceImageGatedCrossAttention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=16, scale=4, dropout=0.):
        super().__init__()
        self.attn_gate = nn.Parameter(torch.Tensor([0.]))
        self.ff_gate = nn.Parameter(torch.Tensor([0.]))

        self.cross_attn = SequenceImageCrossAttention(dim, heads=heads, dim_head=dim_head)
        self.ff = nn.Linear(dim, dim)
    
    def forward(self, text, image_embeds, mask=None):
        out = self.cross_attn(text, image_embeds, mask=mask) * self.attn_gate.tanh() + text
        out = self.ff(out) * self.ff_gate.tanh() + out
        return out
    
class PercieverAttention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=16):
        super().__init__()
        self.scale = dim ** -0.5
        self.heads = heads

        inner_dim = dim * dim_head
        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
    
    def forward(self, tensor, latents):
        q = self.to_q(latents)
        kv_input = torch.cat((tensor, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t : einops.rearrange(t, 'b t n (h d) -> b h t n d', h=self.heads), (q, k, v))
        q = q * self.scale

        sim = torch.einsum('... i d, ... j d  -> ... i j', q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum('... i j, ... j d -> ... i d', attn, v)
        out = einops.rearrange(out, 'b h t n d -> b t n (h d)', h=self.heads)
        return self.to_out(out)
    
class PercieverResampler(nn.Module):
    def __init__(self, dim, depth=3, dim_head=16, heads=2, num_latents=64, num_media_embeds=4, scale=4):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))
        self.norm = nn.LayerNorm(dim)

        self.layers = nn.ModuleList([])
        for layer in range(depth):
            self.layers.append(nn.ModuleList([
                PercieverAttention(dim=dim, dim_head=dim_head, heads=heads),
                nn.Linear(dim, dim, bias=False),
            ]))

    def forward(self, tensor):
        if tensor.ndim == 3:
            tensor = einops.rearrange(tensor, 'b n d -> b 1 n d')

        times = tensor.shape[1]
        tensor = tensor + self.media_pos_emb[:times]

        latents = einops.repeat(self.latents, 'n d -> b m n d', b=tensor.shape[0], m=tensor.shape[1])

        for attn, ff in self.layers:
            latents = attn(tensor, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents).squeeze(1)
    
class ImageSequenceCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 2, dim_head: int = 16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads

        inner_dim = heads * dim_head
        
        self.norm = LayerNorm(dim) 

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, image_embeds: torch.Tensor, text: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        image_embeds_norm = self.norm(image_embeds)
        
        q = self.to_q(image_embeds_norm)
        k, v = self.to_kv(text).chunk(2, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        
        q = q * self.scale
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        if mask is not None:
            expanded_mask = einops.rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~expanded_mask, -torch.finfo(sim.dtype).max)
        
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out