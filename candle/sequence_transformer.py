import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.norm import LayerNorm
from modules.composition import Residual, FeedForward, Identity
from modules.embedding import RotaryEmbedding
from modules.attention import Attend

class TransformerBlock(nn.Module):
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
        use_rope=False,
        use_flash_attn=False,
        is_causal=False,
        max_seq_len=2048,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.use_rope = use_rope

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
        self.rotary_emb = RotaryEmbedding(dim=dim_head, max_seq_len=max_seq_len) if use_rope else None

    def forward(self, x, mask=None):
        # Attention block
        residual = x
        if self.norm_type == 'pre':
            x = self.norm1(x)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.heads, self.dim_head).transpose(1,2), (q, k, v))
        if self.use_rope and self.rotary_emb is not None:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)
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

class SequenceTransformerBody(nn.Module):
    """
    Fully customizable SOTA sequence transformer.

    Args:
        dim (int): Input and output dimension.
        depth (int): Number of transformer blocks.
        heads (int): Number of attention heads.
        dim_head (int): Dimension per attention head.
        ff_mult (float): Feedforward expansion factor.
        attn_dropout (float): Dropout for attention.
        ff_dropout (float): Dropout for feedforward.
        norm_type (str): 'pre' or 'post' normalization.
        activation (callable): Activation function.
        use_rope (bool): Use rotary embeddings.
        use_flash_attn (bool): Use flash attention if available.
        is_causal (bool): Use causal attention mask.
        max_seq_len (int): Maximum sequence length for rotary embeddings.
        norm_eps (float): Epsilon for normalization.
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
        use_rope=False,
        use_flash_attn=False,
        is_causal=False,
        max_seq_len=2048,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                norm_type=norm_type,
                activation=activation,
                use_rope=use_rope,
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

class SequenceTransformer(nn.Module):
    """
    SOTA-ready sequence transformer with embedding and transformer body.

    Args:
        vocab_size (int): Vocabulary size for token embedding.
        dim (int): Model dimension.
        depth (int): Number of transformer blocks.
        heads (int): Number of attention heads.
        dim_head (int): Dimension per attention head.
        ff_mult (float): Feedforward expansion factor.
        attn_dropout (float): Dropout for attention.
        ff_dropout (float): Dropout for feedforward.
        norm_type (str): 'pre' or 'post' normalization.
        activation (callable): Activation function.
        use_rope (bool): Use rotary embeddings.
        use_flash_attn (bool): Use flash attention if available.
        is_causal (bool): Use causal attention mask.
        max_seq_len (int): Maximum sequence length for embeddings.
        norm_eps (float): Epsilon for normalization.
        use_pos_emb (bool): Use learned positional embedding.
    """
    def __init__(
        self,
        vocab_size,
        dim,
        depth=12,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        norm_type='pre',
        activation=nn.GELU,
        use_rope=False,
        use_flash_attn=False,
        is_causal=False,
        max_seq_len=2048,
        norm_eps=1e-5,
        use_pos_emb=True,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        else:
            self.pos_emb = None
        self.body = SequenceTransformerBody(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            norm_type=norm_type,
            activation=activation,
            use_rope=use_rope,
            use_flash_attn=use_flash_attn,
            is_causal=is_causal,
            max_seq_len=max_seq_len,
            norm_eps=norm_eps,
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq) token indices
            mask: (batch, seq) or (batch, 1, seq, seq) attention mask
        """
        x = self.token_emb(x)
        if self.use_pos_emb and self.pos_emb is not None:
            x = x + self.pos_emb[:, :x.size(1), :]
        x = self.body(x, mask=mask)
        return x
