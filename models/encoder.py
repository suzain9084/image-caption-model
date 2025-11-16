import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class EncoderCNN(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()

        self.convSequence = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv2d(128, 128, 3, stride=2),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Conv2d(256, 256, 3, stride=2),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(512),
            nn.Conv2d(512, 512, 3, stride=2),
            nn.ReLU(),
        )

        self.feature_projection = nn.Linear(512, feature_dim)


    def forward(self, x):
        x = self.convSequence(x)

        batch_size, channels, h, w = x.size()

        x = x.view(batch_size, channels, h * w)
        x = x.transpose(1, 2)
        x = self.feature_projection(x)

        return x
