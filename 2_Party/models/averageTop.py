import torch.nn as nn

class TopModel(nn.Module):
    def __init__(self, input_dim=128, num_classes=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ELU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
