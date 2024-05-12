import torch
from torch import nn


class MyCNN(nn.Module):
    def __init__(self, n_classes):
        super(MyCNN, self).__init__()
        self.nn = nn.Sequential(
            nn.LayerNorm((3, 128, 256)),
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.AvgPool2d((2, 4)),  # [8, 64, 64]

            nn.LayerNorm((8, 64, 64)),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # [16, 32, 32]

            nn.LayerNorm((16, 32, 32)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d((4, 4)),  # [32, 8, 8]

            nn.LayerNorm((32, 8, 8)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2)),  # [64, 4, 4]

            nn.Flatten(),
            nn.LayerNorm(64 * 4 * 4),
            nn.Linear(64 * 4 * 4, 128),
            nn.Tanh(),
        )

        self.out_layer = nn.Linear(128, n_classes, bias=True)
        
    def forward(self, x):
        features = self.nn(x)
        
        return self.out_layer(features)
