from torch import nn
import torch

class ImageCaptioningModel(nn.Module):
    """Combined Encoder-Decoder model with support for both custom and pretrained decoders."""

    def __init__(
        self,
        encoder,
        decoder,
        embed_size,
        pretrained_embeddings=None,
        decoder_type="attention",
        vocab_size=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_type = decoder_type

        if decoder_type == "attention":
            if vocab_size is None:
                vocab_size = decoder.linear.out_features
            self.embed_layer = nn.Embedding(vocab_size, embed_size)

            if pretrained_embeddings is not None:
                self.embed_layer.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
                print(f"Loaded pretrained embeddings: {pretrained_embeddings.shape}")
                self.embed_layer.weight.requires_grad = False
        else:
            self.embed_layer = None

    def forward(self, images, captions, attention_mask=None, labels=None):
        features = self.encoder(images)

        if self.decoder_type == "attention":
            embedded = self.embed_layer(captions)
            if embedded.ndim > 3:
                embedded = embedded.squeeze(1)
            return self.decoder(features, embedded)

        if self.decoder_type == "gpt2_prefix":
            return self.decoder(
                features,
                captions,
                attention_mask=attention_mask,
                labels=labels,
            )

        raise ValueError(f"Unsupported decoder_type: {self.decoder_type}")