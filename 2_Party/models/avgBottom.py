import torch.nn as nn

class BottomModel(nn.Module):
    def __init__(self, input_dim, output_dim=100):
        super().__init__()
        hidden_dims = [
            min(600, input_dim * 4),
            min(300, input_dim * 2),
            output_dim               
        ]
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3 if input_dim < 30 else 0.5),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
