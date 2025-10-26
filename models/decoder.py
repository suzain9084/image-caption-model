from torch import nn
import torch

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.3, feature_dim=512):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.init_h = nn.Linear(feature_dim, hidden_size)
        self.init_c = nn.Linear(feature_dim, hidden_size)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        features = features.squeeze(1)
        h0 = self.init_h(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.init_c(features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        lstm_out, _ = self.lstm(captions, (h0, c0))
        
        outputs = self.linear(self.dropout(lstm_out))
        return outputs
