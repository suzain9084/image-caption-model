from torch import nn
import torch
from models.attention import Attention

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.3, feature_dim=512):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        
        # Attention mechanism
        self.attention = Attention(feature_dim, hidden_size)
        
        # LSTM input is now: word embedding + attention context
        self.lstm = nn.LSTM(
            input_size=embed_size + feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Initialize hidden state from average of spatial features
        self.init_h = nn.Linear(feature_dim, hidden_size)
        self.init_c = nn.Linear(feature_dim, hidden_size)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        batch_size = features.size(0)
        seq_len = captions.size(1)
        
        avg_features = features.mean(dim=1)
        h0 = self.init_h(avg_features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.init_c(avg_features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        outputs = []
        hidden = (h0, c0)
        
        for t in range(seq_len - 1):
            word_embed = captions[:, t, :]
            
            h_t = hidden[0][-1]
            context, alpha = self.attention(features, h_t)
            
            lstm_input = torch.cat([word_embed, context], dim=1)
            lstm_input = lstm_input.unsqueeze(1)
            
            lstm_out, hidden = self.lstm(lstm_input, hidden)
           
            output = self.linear(self.dropout(lstm_out.squeeze(1)))
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        
        return outputs
