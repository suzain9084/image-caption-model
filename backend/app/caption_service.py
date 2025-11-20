import io
from pathlib import Path
from typing import Optional

import torch
import yaml
from PIL import Image
from transformers import AutoTokenizer

from models.captionGenerator import ImageCaptioningModel
from models.encoder import EncoderCNN
from models.pretrained_decoder import GPT2PrefixDecoder
from utils.transform import ImageTransforms


class CaptionService:
    """
    Wraps model loading and inference for FastAPI.
    Loads experiment_3 checkpoint and exposes a generate method.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.config_path = Path(config_path or self.project_root / "config" / "config.yaml")
        self.checkpoint_path = Path(
            checkpoint_path
            or self.project_root / "experiments" / "experiment_3" / "checkpoint" / "best_model.pth"
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.config = self._load_config()
        self.data_cfg = self.config.get("data", {})
        self.model_cfg = self.config.get("model", {})
        self.decoder_type = self.model_cfg.get("decoder_type", "gpt2_prefix")
        self.max_len = self.data_cfg.get("max_caption_length", 22)

        self.transforms = ImageTransforms(
            image_size=tuple(self.data_cfg.get("image_size", [256, 256]))
        ).get_test_transforms()

        self.tokenizer = self._load_tokenizer()
        self._load_model()

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as fp:
            return yaml.safe_load(fp)

    def _load_tokenizer(self):
        if not self.data_cfg.get("use_hf_tokenizer", False):
            raise ValueError(
                "The backend currently supports only HF tokenizer-based checkpoints."
            )

        tokenizer_name = self.data_cfg.get("hf_tokenizer_name", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.checkpoint_path}. "
                "Please train the model or download experiment_3 weights."
            )

        if self.decoder_type != "gpt2_prefix":
            raise ValueError(
                f"Inference server expects decoder_type='gpt2_prefix', got {self.decoder_type}"
            )

        encoder = EncoderCNN(
            feature_dim=self.model_cfg.get("feature_dim", 512),
            encoder_name=self.model_cfg.get("encoder_name", "efficientnet_b0"),
            pretrained=self.model_cfg.get("encoder_pretrained", True),
            trainable_blocks=self.model_cfg.get("encoder_trainable_blocks", 1),
        )

        decoder = GPT2PrefixDecoder(
            feature_dim=self.model_cfg.get("feature_dim", 512),
            decoder_model_name=self.model_cfg.get("hf_decoder_name", "gpt2"),
            prefix_length=self.model_cfg.get("decoder_prefix_length", 16),
            freeze_decoder_layers=0, 
        )

        self.model = ImageCaptioningModel(
            encoder=encoder,
            decoder=decoder,
            embed_size=self.model_cfg.get("embed_size", 100),
            decoder_type=self.decoder_type,
        ).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.model.eval()

    def _preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
        tensor = self.transforms(image).unsqueeze(0).to(self.device)
        return tensor

    def generate_caption(self, image_bytes: bytes) -> str:
        image_tensor = self._preprocess_image(image_bytes)
        with torch.no_grad():
            features = self.model.encoder(image_tensor)
            input_ids = torch.tensor(
                [[self.tokenizer.bos_token_id]], device=self.device, dtype=torch.long
            )
            attention_mask = torch.ones_like(input_ids, device=self.device)

            for _ in range(self.max_len):
                decoder_outputs = self.model.decoder(
                    features,
                    input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                )

                logits = decoder_outputs["logits"]
                prefix_len = decoder_outputs.get("prefix_length", 0)
                seq_len = decoder_outputs.get("sequence_length", input_ids.size(1))
                next_token_logits = logits[:, prefix_len + seq_len - 1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_token_id], dim=1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((attention_mask.size(0), 1), device=self.device, dtype=torch.long),
                    ],
                    dim=1,
                )

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

        caption_tokens = input_ids[0, 1:]  # drop BOS
        caption = self.tokenizer.decode(
            caption_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return caption.strip()

