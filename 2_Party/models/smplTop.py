import torch.nn as nn

class TopModel(nn.Module):
    def __init__(self, input_dim=200, num_classes=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),  
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)
