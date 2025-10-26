import torch
from torch import nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.convSequence = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.linearSequence = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),

            nn.Linear(1024, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.convSequence(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linearSequence(x)
        return x
