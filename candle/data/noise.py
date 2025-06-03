import torch
import torch.nn.functional as F
from tqdm import tqdm

class NoiseScheduler:
    def __init__(self, device, timesteps=300):
        self.device = device
        self.timesteps = 300
        self.betas = self.linear_beta_schedule(timesteps=timesteps)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps, device=self.device)
    
    def extract(self, alphas, tensor, shape):
        batch_size = tensor.shape[0]
        out = alphas.gather(-1, tensor)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(tensor.device)
    
    def q_sample(self, image, timestep, noise=None):
        if type(timestep) == int:
            timestep = torch.Tensor([timestep]).long()
        if noise is None:
            noise = torch.randn_like(image)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, timestep, image.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, timestep, image.shape
        )

        return sqrt_alphas_cumprod_t * image + sqrt_one_minus_alphas_cumprod_t * noise

    def __call__(self, image, timesteps, noise=None):
        return self.q_sample(image, timesteps, noise=noise)