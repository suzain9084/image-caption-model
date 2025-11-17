import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from data.processed.data_processed import CaptionDataset
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN
from models.captionGenerator import ImageCaptioningModel
from utils.transform import ImageTransforms
from utils.metrics import evaluate_caption_metrics


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_dataloader(config, split: str = "val"):
    image_transforms = ImageTransforms(
        image_size=tuple(config["data"]["image_size"])
    )

    if split == "train":
        transform = image_transforms.get_train_transforms()
        annotation_path = config["data"]["train_annotation_path"]
        image_path = config["data"]["train_image_path"]
    else:
        transform = image_transforms.get_val_transforms()
        annotation_path = config["data"]["val_annotation_path"]
        image_path = config["data"]["val_image_path"]

    dataset = CaptionDataset(
        annotation_path=annotation_path,
        image_path=image_path,
        transform=transform,
        max_len=config["data"]["max_caption_length"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=False,
    )

    return dataloader, dataset


def build_model(config, vocab_size: int, device: torch.device, vocabulary, use_pretrained: bool):
    """Create encoder, decoder and combined ImageCaptioningModel.

    vocab_size is taken from the checkpoint (decoder linear layer) so that
    the architecture exactly matches the trained model.
    """
    encoder = EncoderCNN(
        feature_dim=config["model"]["feature_dim"]
    )

    decoder = DecoderRNN(
        embed_size=config["model"]["embed_size"],
        hidden_size=config["model"]["hidden_size"],
        vocab_size=vocab_size,
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        feature_dim=config["model"]["feature_dim"],
    )

    # Optional pretrained embeddings (same logic as in training)
    pretrained_embeddings = None
    if use_pretrained:
        glove_path = config["model"]["glove_path"]
        embedding_dim = config["model"]["embed_size"]
        print(f"Loading GloVe embeddings from {glove_path}...")
        pretrained_embeddings = vocabulary.load_pretrained_embeddings(
            glove_path, embedding_dim
        )
        print(f"Pretrained embeddings shape: {pretrained_embeddings.shape}")

    model = ImageCaptioningModel(
        encoder,
        decoder,
        config["model"]["embed_size"],
        pretrained_embeddings=pretrained_embeddings,
    )

    model = model.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    return model, encoder, decoder


def load_checkpoint(checkpoint_path: str, device: torch.device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print("Checkpoint loaded successfully.")
    return checkpoint


def main(config_path: str, max_samples: int | None = None):
    config = load_config(config_path)

    # Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and config["training"]["use_cuda"]
        else "cpu"
    )
    print(f"Using device: {device}")

    # Rebuild vocabulary from TRAIN annotations (to match training time)
    _, train_dataset = build_dataloader(config, split="train")
    vocabulary = train_dataset.vocabulary
    train_vocab_size = vocabulary.vocab_size()
    print(f"Vocabulary size from training dataset: {train_vocab_size}")

    # Validation data loader (will reuse train vocabulary for decoding)
    val_loader, val_dataset = build_dataloader(config, split="val")
    # Important: use the training vocabulary for validation captions too
    val_dataset.vocabulary = vocabulary

    # Load checkpoint from experiment_1 (or path in config) to get trained vocab size
    checkpoint_dir = config["training"]["checkpoint_dir"]
    checkpoint_path = str(Path(checkpoint_dir) / "best_model.pth")
    checkpoint = load_checkpoint(checkpoint_path, device)

    if "decoder_state_dict" not in checkpoint:
        raise KeyError("decoder_state_dict not found in checkpoint.")

    decoder_state = checkpoint["decoder_state_dict"]
    # Decoder output dimension (vocab size during training)
    ckpt_vocab_size = decoder_state["linear.weight"].shape[0]
    print(f"Vocabulary size from checkpoint decoder: {ckpt_vocab_size}")

    # Only use pretrained embeddings if vocab size matches; otherwise shapes will differ
    use_pretrained = (
        config["model"].get("use_pretrained_embeddings", False)
        and ckpt_vocab_size == train_vocab_size
    )
    if config["model"].get("use_pretrained_embeddings", False) and not use_pretrained:
        print(
            "Warning: vocab size from checkpoint and current dataset differ; "
            "skipping pretrained embeddings for evaluation."
        )

    # Build model with vocab size taken from checkpoint
    model, encoder, decoder = build_model(
        config, ckpt_vocab_size, device, vocabulary, use_pretrained
    )

    # Load encoder/decoder weights
    if "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(decoder_state)

    # Run caption metrics
    print("\n" + "=" * 50)
    print("Evaluating caption metrics on validation set (experiment 1)")
    print("=" * 50)

    scores = evaluate_caption_metrics(
        model,
        encoder,
        decoder,
        val_loader,
        vocabulary,
        device=device,
        max_samples=max_samples,
        max_len=config["data"]["max_caption_length"],
    )

    print("\nCaption Evaluation Metrics:")
    print(f"  BLEU-1 : {scores['bleu1']:.4f}")
    print(f"  BLEU-4 : {scores['bleu4']:.4f}")
    print(f"  METEOR : {scores['meteor']:.4f}")
    print(f"  ROUGE-L: {scores['rouge_l']:.4f}")
    print(f"  CIDEr  : {scores['cider']:.4f}")
    print(f"  SPICE  : {scores['spice']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Image Captioning Model (Experiment 1)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of validation batches to evaluate",
    )
    args = parser.parse_args()

    main(args.config, args.max_samples)


