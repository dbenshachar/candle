import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def forward(self, tensor):
        tensor, gate = tensor.chunk(2, dim=-1)
        return F.silu(gate) * tensor
    
class GEGLU(nn.Module):
    def forward(self, tensor):
        tensor, gate = tensor.chunk(2, dim=-1)
        return tensor * F.gelu(gate)