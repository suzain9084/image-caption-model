import torch
from torch import nn
from transformers import AutoModelForCausalLM


class GPT2PrefixDecoder(nn.Module):
    """
    Decoder that reuses a pretrained GPT-2 language model and prepends a small
    sequence of learned tokens derived from the image encoder. This lets us keep
    the custom encoder while benefitting from a powerful text decoder.
    """

    def __init__(
        self,
        feature_dim: int,
        decoder_model_name: str = "gpt2",
        prefix_length: int = 16,
        freeze_decoder_layers: int = 0,
    ):
        super().__init__()
        self.prefix_length = prefix_length
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name)
        self.hidden_size = self.decoder.config.hidden_size

        # Map encoder features into the GPT-2 hidden space and expand to prefix tokens.
        self.prefix_mapper = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.hidden_size * prefix_length),
            nn.Tanh(),
        )

        if freeze_decoder_layers > 0:
            self._freeze_decoder_layers(freeze_decoder_layers)

    def _freeze_decoder_layers(self, trainable_layers: int) -> None:
        transformer = getattr(self.decoder, "transformer", None)
        if transformer is None:
            return

        blocks = getattr(transformer, "h", [])
        total_layers = len(blocks)
        trainable_layers = min(trainable_layers, total_layers)
        trainable_start = total_layers - trainable_layers

        for idx, block in enumerate(blocks):
            requires_grad = idx >= trainable_start
            for param in block.parameters():
                param.requires_grad = requires_grad

        # Keep layer norm, embeddings, and LM head trainable for adaptation.
        for param in transformer.ln_f.parameters():
            param.requires_grad = True
        for param in transformer.wte.parameters():
            param.requires_grad = True
        for param in self.decoder.lm_head.parameters():
            param.requires_grad = True

    def forward(
        self,
        features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        batch_size = features.size(0)
        device = features.device

        pooled_features = features.mean(dim=1)
        prefix_tokens = self.prefix_mapper(pooled_features).view(
            batch_size, self.prefix_length, self.hidden_size
        )

        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_tokens, inputs_embeds], dim=1)

        if attention_mask is not None:
            prefix_mask = torch.ones(
                batch_size, self.prefix_length, dtype=attention_mask.dtype, device=device
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        if labels is not None:
            prefix_labels = labels.new_full((batch_size, self.prefix_length), -100)
            labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
            "prefix_length": self.prefix_length,
            "sequence_length": input_ids.size(1),
        }

