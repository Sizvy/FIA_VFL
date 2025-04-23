import torch.nn as nn

class BottomModel(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GroupNorm(4, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(0.4),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GroupNorm(4, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)
