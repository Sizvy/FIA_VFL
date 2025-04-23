import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    """Standalone discriminator to distinguish real vs shuffled passive embeddings"""
    def __init__(self, emb_dim):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(emb_dim * 2, 256)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            spectral_norm(nn.Linear(128, 1)),
            nn.Sigmoid()
        )
    
    def forward(self, emb_active, emb_passive):
        combined = torch.cat([emb_active, emb_passive], dim=1)
        return self.model(combined)
