import torch.nn as nn

class EnhancedDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256)
        )
        
        # Discriminator head (real/fake)
        self.disc_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Feature value predictor head
        self.reg_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        features = self.shared(x)
        is_real = self.disc_head(features)
        pred_value = self.reg_head(features)
        return is_real, pred_value.squeeze()
