from torch import nn
import torch

class ImageCaptioningModel(nn.Module):
    """Combined Encoder-Decoder model with Attention"""
    def __init__(self, encoder, decoder, embed_size, pretrained_embeddings=None):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        vocab_size = decoder.linear.out_features
        self.embed_layer = nn.Embedding(vocab_size, embed_size)
        
        if pretrained_embeddings is not None:
            self.embed_layer.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            print(f"Loaded pretrained embeddings: {pretrained_embeddings.shape}")
        
    def forward(self, images, captions):
        features = self.encoder(images)
        
        embedded_captions = self.embed_layer(captions)
        embedded_captions = embedded_captions.squeeze(1) if len(embedded_captions.shape) > 3 else embedded_captions
        
        outputs = self.decoder(features, embedded_captions)
        return outputs