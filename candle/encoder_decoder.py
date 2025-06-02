import torch
import torch.nn as nn

from .vision_transformer import VisionTransformerBody
from .modules.vq import VectorQuantizeEMA

class EncoderDecoder(nn.Module):
    """
    Encoder-Decoder architecture with VisionTransformerBody encoder, vector quantization, and transformer decoder.

    Args:
        image_size (int): Input image size.
        patch_size (int): Patch size for encoder.
        in_channels (int): Number of input channels.
        encoder_dim (int): Encoder model dimension.
        encoder_depth (int): Encoder transformer depth.
        vq_num_embeddings (int): Number of VQ codebook vectors.
        vq_embedding_dim (int): VQ embedding dimension.
        decoder_dim (int): Decoder model dimension.
        decoder_depth (int): Decoder transformer depth.
        out_dim (int): Output dimension (e.g., number of classes or channels).
        ... (other transformer args as needed)
    """
    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        encoder_dim,
        encoder_depth,
        vq_num_embeddings,
        vq_embedding_dim,
        decoder_dim,
        decoder_depth,
        out_dim,
        encoder_heads=8,
        encoder_dim_head=64,
        decoder_heads=8,
        decoder_dim_head=64,
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
        # Encoder: Vision Transformer Body
        self.encoder = VisionTransformerBody(
            dim=encoder_dim,
            depth=encoder_depth,
            heads=encoder_heads,
            dim_head=encoder_dim_head,
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
        # Patch embedding for encoder
        self.patch_emb = nn.Conv2d(in_channels, encoder_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2

        # Vector Quantization
        self.vq = VectorQuantizeEMA(
            num_embeddings=vq_num_embeddings,
            embedding_dim=vq_embedding_dim,
        )
        # Project encoder output to VQ embedding dim if needed
        if encoder_dim != vq_embedding_dim:
            self.encoder_to_vq = nn.Linear(encoder_dim, vq_embedding_dim)
        else:
            self.encoder_to_vq = nn.Identity()

        # Decoder: simple transformer body
        self.decoder = VisionTransformerBody(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
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
        # Project VQ output to decoder dim if needed
        if vq_embedding_dim != decoder_dim:
            self.vq_to_decoder = nn.Linear(vq_embedding_dim, decoder_dim)
        else:
            self.vq_to_decoder = nn.Identity()

        # Output head
        self.head = nn.Linear(decoder_dim, out_dim)

    def encode(self, x):
        """
        Encode input image to quantized latent representation.
        Args:
            x: (batch, in_channels, image_size, image_size)
        Returns:
            quantized: (batch, num_patches, vq_embedding_dim)
            vq_info: dict with VQ stats
        """
        x = self.patch_emb(x)  # (batch, encoder_dim, h, w)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, encoder_dim)
        x = self.encoder(x)  # (batch, num_patches, encoder_dim)
        x = self.encoder_to_vq(x)  # (batch, num_patches, vq_embedding_dim)
        # Add dummy dimension for VQ (b n j d) -> (b n 1 d)
        x = x.unsqueeze(2)
        vq_out = self.vq(x)
        quantized = vq_out['quantized'].squeeze(2)  # (batch, num_patches, vq_embedding_dim)
        return quantized, vq_out

    def decode(self, quantized):
        """
        Decode quantized latent to output.
        Args:
            quantized: (batch, num_patches, vq_embedding_dim)
        Returns:
            output: (batch, out_dim)
        """
        x = self.vq_to_decoder(quantized)  # (batch, num_patches, decoder_dim)
        x = self.decoder(x)  # (batch, num_patches, decoder_dim)
        x = x.mean(dim=1)  # global average pooling
        return self.head(x)

    def forward(self, x):
        """
        Full encode-decode forward.
        Args:
            x: (batch, in_channels, image_size, image_size)
        Returns:
            output: (batch, out_dim)
            vq_info: dict with VQ stats
        """
        quantized, vq_info = self.encode(x)
        output = self.decode(quantized)
        return output, vq_info
