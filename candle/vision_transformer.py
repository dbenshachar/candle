import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.norm import LayerNorm
from .modules.composition import FeedForward, Identity
from .modules.embedding import SinusoidalPositionalEmbedding
from .modules.attention import Attend

class VisionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        norm_type='pre',
        activation=nn.GELU,
        use_flash_attn=False,
        is_causal=False,
        max_seq_len=2048,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.norm_type = norm_type

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.attend = Attend(
            heads=heads,
            dropout=attn_dropout,
            use_flash_attn=use_flash_attn,
            is_casual=is_causal,
        )
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, dim, bias=False)
        self.ff = FeedForward(
            dim,
            activation=activation,
            norm=None,
            scale=ff_mult,
            dropout=ff_dropout,
        )

    def forward(self, x, mask=None):
        # Attention block
        residual = x
        if self.norm_type == 'pre':
            x = self.norm1(x)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.heads, self.dim_head).transpose(1,2), (q, k, v))
        q = q * self.scale
        attn_out = self.attend(q, k, v, mask=mask)
        attn_out = attn_out.transpose(1,2).contiguous().view(x.shape[0], x.shape[1], -1)
        attn_out = self.to_out(attn_out)
        x = residual + attn_out
        if self.norm_type == 'post':
            x = self.norm1(x)

        # Feedforward block
        residual = x
        if self.norm_type == 'pre':
            x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        if self.norm_type == 'post':
            x = self.norm2(x)
        return x

class VisionTransformerBody(nn.Module):
    """
    Stack of VisionTransformerBlocks for vision transformer models.

    Args:
        dim (int): Input/output dimension.
        depth (int): Number of transformer blocks.
        ... (see VisionTransformerBlock for other args)
    """
    def __init__(
        self,
        dim,
        depth=12,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        norm_type='pre',
        activation=nn.GELU,
        use_flash_attn=False,
        is_causal=False,
        max_seq_len=2048,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            VisionTransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                norm_type=norm_type,
                activation=activation,
                use_flash_attn=use_flash_attn,
                is_causal=is_causal,
                max_seq_len=max_seq_len,
                norm_eps=norm_eps,
            )
            for _ in range(depth)
        ])
        self.final_norm = LayerNorm(dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.final_norm(x)

class VisionTransformer(nn.Module):
    """
    SOTA-ready vision transformer with patch embedding and transformer body.

    Args:
        image_size (int): Input image size (assumes square images).
        patch_size (int): Patch size.
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        dim (int): Model dimension.
        depth (int): Number of transformer blocks.
        ... (see VisionTransformerBody for other args)
        use_cls_token (bool): Use class token.
        use_pos_emb (bool): Use learned positional embedding.
    """
    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        num_classes,
        dim,
        depth=12,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        norm_type='pre',
        activation=nn.GELU,
        use_flash_attn=False,
        is_causal=False,
        max_seq_len=2048,
        norm_eps=1e-5,
        use_cls_token=True,
        use_pos_emb=True,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by patch size."
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.use_cls_token = use_cls_token
        self.use_pos_emb = use_pos_emb

        self.patch_emb = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        else:
            self.cls_token = None
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + (1 if use_cls_token else 0), dim))
        else:
            self.pos_emb = None

        self.body = VisionTransformerBody(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            norm_type=norm_type,
            activation=activation,
            use_flash_attn=use_flash_attn,
            is_causal=is_causal,
            max_seq_len=max_seq_len,
            norm_eps=norm_eps,
        )
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, in_channels, image_size, image_size)
        """
        x = self.patch_emb(x)  # (batch, dim, h, w)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, dim)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        if self.use_pos_emb and self.pos_emb is not None:
            x = x + self.pos_emb[:, :x.size(1), :]
        x = self.body(x, mask=mask)
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        return self.head(x)
