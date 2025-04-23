import torch.nn as nn

class TopModel(nn.Module):
    def __init__(self, input_dim=128, num_classes=11):
        """More complex top model with additional layers and regularization"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GroupNorm(4, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.GroupNorm(4, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
