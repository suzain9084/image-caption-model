import torch
from torch import nn
import math

class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attn_dim=256):
        super().__init__()
        self.attn_dim = attn_dim
        self.W_h = nn.Linear(hidden_dim, attn_dim)
        self.W_a = nn.Linear(feature_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.xavier_uniform_(self.W_a.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, features, hidden):
        h_proj = self.W_h(hidden).unsqueeze(1)
        a_proj = self.W_a(features)
        # Use attn_dim for normalization, not a_proj.size(-1)
        energy = self.v(torch.tanh(h_proj + a_proj)).squeeze(2) / math.sqrt(self.attn_dim)
        alpha = torch.softmax(energy, dim=1)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha