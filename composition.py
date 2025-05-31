import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, tensor):
        return self.function(tensor) + tensor
    
class Identity(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()

  def forward(self, tensor):
    return tensor
    
class FeedForward(nn.Module):
    def __init__(self, dim, activation=nn.ReLU, norm=None, scale=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * scale)

        if norm is None:
          norm = Identity()

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias = False),
            activation(),
            norm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(self, tensor):
        return self.net(tensor)