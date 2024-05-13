import torch
from torch import nn


class CNNClassifier(nn.Module):
    def __init__(self, n_classes):
        super(CNNClassifier, self).__init__()
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


class CNNRecognize(nn.Module):
    def __init__(self, input_channel, image_size, classes=2):
        self.classes = classes
        if classes == 2:
            output_channel = 1
        else:
            output_channel = classes
        throughput_size = (input_channel, image_size[0], image_size[1])
        super(CNNRecognize, self).__init__()
        self.nn = nn.Sequential(
            nn.LayerNorm(throughput_size),
            nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(),
            nn.Conv2d(input_channel, input_channel, kernel_size=3, padding=3, dilation=3),
            nn.LayerNorm(throughput_size),
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=3, dilation=3),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.nn(x)
