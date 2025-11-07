from torch import nn
import torch

class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_a = nn.Linear(feature_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, features, hidden):
        hidden_expanded = self.W_h(hidden).unsqueeze(1)
        scores = self.v(torch.tanh(hidden_expanded + self.W_a(features))).squeeze(2)
        alpha = torch.softmax(scores, dim=1)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)

        return context, alpha