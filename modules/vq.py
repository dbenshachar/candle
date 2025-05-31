import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

class VectorQuantizeEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, commit_cost=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.decay = decay
        self.commit_cost = commit_cost
        self.epsilon = 1e-10

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()

        self.ema_cluster_size =  torch.zeros(num_embeddings)
        self.ema_weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.ema_weight.data.normal_()

    def forward(self, latent):
        latent = einops.rearrange(latent, 'b n j d -> b j d n').contiguous()
        latent_flattened = latent.view(-1, self.embedding_dim)

        distances = torch.sum(latent_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * (torch.matmul(latent_flattened, self.embedding.weight.T))

        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings).to(latent.device)
        encodings.scatter_(1, min_encoding_indices, 1)

        latent_quantized = torch.matmul(encodings, self.embedding.weight).view(latent.shape)

        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 0)
            
            ema_sum = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = ((self.ema_cluster_size + self.epsilon) / (ema_sum + self.num_embeddings * self.epsilon) * ema_sum)
            
            self.ema_weight = nn.Parameter(self.ema_weight * self.decay + (1 - self.decay) * torch.matmul(encodings.T, latent_flattened))
            
            self.embedding.weight = nn.Parameter(self.ema_weight / self.ema_cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(latent_quantized.detach(), latent)
        loss = self.commit_cost * e_latent_loss

        latent_quantized = latent + (latent_quantized - latent).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        latent_quantized = einops.rearrange(latent_quantized, 'b j d n -> b n j d').contiguous()

        return {'quantized' : latent_quantized, 'loss' : loss, 'perplexity' : perplexity, 'encodings' : encodings, 'distances': distances}