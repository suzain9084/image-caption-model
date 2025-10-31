from torch import nn
import torch

class ImageCaptioningModel(nn.Module):
    """Combined Encoder-Decoder model"""
    def __init__(self, encoder, decoder, embed_size, pretrained_embeddings=None):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        # Create embedding layer
        vocab_size = decoder.linear.out_features
        self.embed_layer = nn.Embedding(vocab_size, embed_size)
        
        # Load pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embed_layer.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            print(f"Loaded pretrained embeddings: {pretrained_embeddings.shape}")
        
    def forward(self, images, captions):
        features = self.encoder(images)
        features = features.unsqueeze(1)
        
        embedded_captions = self.embed_layer(captions)
        embedded_captions = embedded_captions.squeeze(1)
        outputs = self.decoder(features, embedded_captions)
        return outputs