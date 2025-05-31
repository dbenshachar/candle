import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import Downsample, Upsample, UNetBlock
from modules.norm import ChanLayerNorm
from modules.embedding import SinusoidalTimeEmbedding
from modules.attention import SequenceImageCrossAttention
from modules.composition import FeedForward
from modules.embedding import SinusoidalPositionalEmbedding

class Text2ImageUNet(nn.Module):
    """
    U-Net with SequenceImageCrossAttention for text-to-image diffusion.
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
        text_dim=768,
        cross_attn_heads=4,
        cross_attn_dim_head=32,
        use_time=True,
        max_seq_len=128,
    ):
        super().__init__()
        self.use_time = use_time
        if use_time:
            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbedding(base_channels),
                nn.Linear(base_channels, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.time_mlp = None

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
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

        self.mid = nn.Sequential(
            UNetBlock(in_ch, in_ch, time_emb_dim, groups=groups),
            UNetBlock(in_ch, in_ch, time_emb_dim, groups=groups),
        )

        self.cross_attn = SequenceImageCrossAttention(
            dim=text_dim,
            heads=cross_attn_heads,
            dim_head=cross_attn_dim_head,
        )
        self.text_pos_emb = SinusoidalPositionalEmbedding(text_dim, max_seq_len=max_seq_len)

        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(UNetBlock(in_ch + channels.pop(), out_ch, time_emb_dim, groups=groups))
                in_ch = out_ch
            self.upsamples.append(Upsample(in_ch))

        self.final_norm = ChanLayerNorm(in_ch)
        self.final_conv = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, x, t=None, text_emb=None, text_mask=None):
        """
        Args:
            x: (batch, in_channels, height, width)
            t: (batch,) time steps or None if use_time=False
            text_emb: (batch, seq, text_dim)
            text_mask: (batch, seq) boolean mask
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

        # Cross-attention: let image attend to text
        if text_emb is not None:
            text_emb = self.text_pos_emb(text_emb)
            # flatten spatial dims for cross-attn
            b, c, h_img, w_img = h.shape
            img_tokens = h.view(b, c, h_img * w_img).permute(0, 2, 1)  # (b, hw, c)
            img_tokens = self.cross_attn(img_tokens, text_emb, mask=text_mask)
            h = img_tokens.permute(0, 2, 1).view(b, c, h_img, w_img)

        # Up path
        for up, upsample in zip(self.ups, self.upsamples):
            h = torch.cat([h, hs.pop()], dim=1)
            h = up(h, t_emb)
            h = upsample(h)

        h = self.final_norm(h)
        return self.final_conv(h)

    @torch.no_grad()
    def generate(self, text_emb, text_mask=None, shape=(1, 3, 32, 32), steps=50, device=None):
        """
        Simple DDIM-like sampling loop for text-to-image.
        """
        device = device or next(self.parameters()).device
        x = torch.randn(shape, device=device)
        t_seq = torch.linspace(1, 0, steps, device=device)
        for t in t_seq:
            t_batch = torch.full((shape[0],), t, device=device)
            x = self.forward(x, t_batch, text_emb, text_mask)
        return x

class Image2TextAutoregressive(nn.Module):
    """
    Autoregressive transformer for generating text from images using cross-attention.
    """
    def __init__(
        self,
        image_dim,
        vocab_size,
        seq_len,
        text_dim=768,
        num_layers=8,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        norm_type='pre',
        activation=nn.GELU,
        cross_attn_heads=4,
        cross_attn_dim_head=32,
        max_seq_len=128,
    ):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, text_dim)
        self.token_emb = nn.Embedding(vocab_size, text_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, text_dim))
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=text_dim,
                nhead=heads,
                dim_feedforward=int(text_dim * ff_mult),
                dropout=ff_dropout,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.cross_attn = SequenceImageCrossAttention(
            dim=text_dim,
            heads=cross_attn_heads,
            dim_head=cross_attn_dim_head,
        )
        self.head = nn.Linear(text_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, image_emb, tokens, image_mask=None, text_mask=None):
        """
        Args:
            image_emb: (batch, image_tokens, image_dim)
            tokens: (batch, seq_len) token indices
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        image_emb = self.image_proj(image_emb)
        x = self.token_emb(tokens)
        x = x + self.pos_emb[:, :x.size(1), :]
        # Cross-attend image features to text tokens
        x = self.cross_attn(x, image_emb, mask=image_mask)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, image_emb, bos_token_id, eos_token_id=None, max_len=None, image_mask=None, device=None):
        """
        Autoregressive text generation from image embedding.
        """
        device = device or next(self.parameters()).device
        max_len = max_len or self.seq_len
        batch_size = image_emb.size(0)
        tokens = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            logits = self.forward(image_emb, tokens, image_mask=image_mask)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break
        return tokens
