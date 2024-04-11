import torch
from torch import nn


class MyCNN(nn.Module):
    def __init__(self, n_classes):
        super(MyCNN, self).__init__()
        self.nn = nn.Sequential(
            nn.LayerNorm((1, 64, 64)),
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2)),  # [8, 32, 32]

            nn.LayerNorm((8, 32, 32)),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # [16, 16, 16]

            nn.LayerNorm((16, 16, 16)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # [8, 8, 32]

            nn.LayerNorm((32, 8, 8)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2)),  # [4, 4, 64]

            nn.Flatten(),
            nn.LayerNorm(4 * 4 * 64),
            nn.Linear(4 * 4 * 64, 128),
            nn.Tanh(),
        )

        self.out_layer = nn.Linear(128, n_classes, bias=False)

    def forward(self, x):

        features = self.nn(x)
        weights = self.out_layer.weight.T

        #features = nn.functional.normalize(features, dim=-1)
        #weights = nn.functional.normalize(weights, dim=-1)

        out = torch.matmul(features, weights)

        return out
