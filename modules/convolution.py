import torch
from torch import nn
from torch.nn import functional as F
import einops.layers.torch as layers
from norm import ChanLayerNorm

class ShiftPoolingTransformer(nn.Module):
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
  if out_channels is None:
    out_channels = channels
  return nn.Conv2d(channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels, stride=1)