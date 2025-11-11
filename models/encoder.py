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

        self.feature_projection = nn.Linear(512, feature_dim)

    def forward(self, x):
        x = self.convSequence(x)
        
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width)
        x = x.transpose(1, 2)
        drop_out = nn.Dropout(0.3)
        x = drop_out(x)
        x = self.feature_projection(x)
        
        return x
