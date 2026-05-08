import torch
import torch.nn as nn

class FusionMLP(nn.Module):
    def __init__(self):
        super(FusionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
            # No Sigmoid — BCEWithLogitsLoss handles it during training
            # torch.sigmoid() applied manually at inference
        )

    def forward(self, x):
        return self.network(x)