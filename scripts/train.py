import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path
# import re
# import json
# import shutil
# import matplotlib.pyplot as plt

from data.processed.data_processed import CaptionDataset
from models.decoder import DecoderRNN
from models.encoder import EncoderCNN
from utils.transform import ImageTransforms
# from utils.metrics import evaluate_caption_metrics
from utils.helper_function import _compute_token_accuracy
from models.captionGenerator import ImageCaptioningModel
# from utils.helper_function import plotAccuracyGraph

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def train_epoch(model, dataloader, criterion, optimizer, device, config):
    model.train()
    total_loss = 0
    total_correct = 0
    total_valid = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, captions) in enumerate(progress_bar):
        images = images.to(device)
        captions = captions.to(device)
        optimizer.zero_grad()
        
        outputs = model(images, captions)
        outputs = outputs.reshape(-1, outputs.shape[2])
        targets = captions.reshape(-1)[1:]
        
        loss = criterion(outputs, targets)
        
        loss.backward()
        
        if config['training']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        
        optimizer.step()
        
        total_loss += loss.item()
        with torch.no_grad():
            batch_acc = _compute_token_accuracy(outputs, targets, ignore_index=0)
            # accumulate per-batch using valid tokens
            valid_mask = targets != 0
            total_correct += batch_acc * valid_mask.sum().item()
            total_valid += valid_mask.sum().item()
        
        progress_bar.set_postfix({'loss': loss.item(), 'acc': batch_acc})
        
    epoch_acc = (total_correct / max(total_valid, 1)) if total_valid > 0 else 0.0
    return total_loss / len(dataloader), epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_valid = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for images, captions in progress_bar:
            images = images.to(device)
            captions = captions.to(device)
            
            outputs = model(images, captions)
            
            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = captions.reshape(-1)[1:]
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            batch_acc = _compute_token_accuracy(outputs, targets, ignore_index=0)
            valid_mask = targets != 0
            total_correct += batch_acc * valid_mask.sum().item()
            total_valid += valid_mask.sum().item()
            
            progress_bar.set_postfix({'loss': loss.item(), 'acc': batch_acc})
    
    epoch_acc = (total_correct / max(total_valid, 1)) if total_valid > 0 else 0.0
    return total_loss / len(dataloader), epoch_acc


def main(config_path):
    config = load_config(config_path)
    
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['use_cuda'] else 'cpu')
    print(f"Using device: {device}")
    
    image_transforms = ImageTransforms(
        image_size=tuple(config['data']['image_size'])
    )
    train_transform = image_transforms.get_train_transforms()
    val_transform = image_transforms.get_val_transforms()
    
    print("Loading datasets...")
    train_dataset = CaptionDataset(
        annotation_path=config['data']['train_annotation_path'],
        image_path=config['data']['train_image_path'],
        transform=train_transform,
        max_len=config['data']['max_caption_length']
    )
    
    val_dataset = CaptionDataset(
        annotation_path=config['data']['val_annotation_path'],
        image_path=config['data']['val_image_path'],
        transform=val_transform,
        max_len=config['data']['max_caption_length']
    )
    
    vocab_size = train_dataset.vocabulary.vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Load pretrained embeddings if specified
    pretrained_embeddings = None
    if config['model'].get('use_pretrained_embeddings', False):
        glove_path = config['model']['glove_path']
        embedding_dim = config['model']['embed_size']
        print(f"Loading GloVe embeddings from {glove_path}...")
        pretrained_embeddings = train_dataset.vocabulary.load_pretrained_embeddings(
            glove_path, embedding_dim
        )
        print(f"Pretrained embeddings shape: {pretrained_embeddings.shape}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=False
    )
    
    print("Initializing models...")
    encoder = EncoderCNN(
        feature_dim=config['model']['feature_dim']
    )
    decoder = DecoderRNN(
        embed_size=config['model']['embed_size'],
        hidden_size=config['model']['hidden_size'],
        vocab_size=vocab_size,
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        feature_dim=config['model']['feature_dim']
    )
    
    model = ImageCaptioningModel(
        encoder, 
        decoder, 
        config['model']['embed_size'],
        pretrained_embeddings=pretrained_embeddings
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Prepare experiment directory: experiments/experiment_i
    experiment_dir = Path("experiments") / f"experiment_{config['training']['experiment_name']}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = experiment_dir / "plots"
    logs_dir = experiment_dir / "logs"

    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training for {config['training']['num_epochs']} epochs...")

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    caption_metrics_history = []  # Store caption metrics for each epoch
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{config['training']['num_epochs']}]")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config)
        print(f"Training Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        checkpoint = {
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'config': config
        }
        save_checkpoint(
            checkpoint,
            config['training']['checkpoint_dir'],
            'best_model.pth'
        )
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")
    
    # Final full evaluation on best model
    # print("\n" + "="*50)
    # print("Computing final caption metrics on full validation set...")
    # print("="*50)
    # try:
    #     # Reload best model if needed (it's already loaded)
    #     final_caption_metrics = evaluate_caption_metrics(
    #         model, encoder, decoder, val_loader, val_dataset.vocabulary,
    #         device, max_samples=None, max_len=config['data']['max_caption_length']
    #     )
    #     print(f"\nFinal Caption Metrics:")
    #     print(f"  BLEU-1: {final_caption_metrics['bleu1']:.4f}")
    #     print(f"  BLEU-4: {final_caption_metrics['bleu4']:.4f}")
    #     print(f"  METEOR: {final_caption_metrics['meteor']:.4f}")
    #     print(f"  ROUGE-L: {final_caption_metrics['rouge_l']:.4f}")
    #     print(f"  CIDEr: {final_caption_metrics['cider']:.4f}")
    #     print(f"  SPICE: {final_caption_metrics['spice']:.4f}")
    # except Exception as e:
    #     print(f"Error computing final caption metrics: {e}")
    #     final_caption_metrics = {}

    # # Save metrics to JSON
    # metrics = {
    #     'train_loss': train_losses,
    #     'val_loss': val_losses,
    #     'train_acc': train_accuracies,
    #     'val_acc': val_accuracies,
    #     'caption_metrics_history': caption_metrics_history,
    #     'final_caption_metrics': final_caption_metrics,
    #     'epochs': list(range(start_epoch + 1, start_epoch + 1 + len(train_losses)))
    # }
    # with open(experiment_dir / 'metrics.json', 'w') as f:
    #     json.dump(metrics, f, indent=2)

    # Plot and save curves
    # plotAccuracyGraph(metrics,train_losses, val_losses, plots_dir, train_accuracies, val_accuracies)

    print(f"\nTraining completed! Experiment saved to: {experiment_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Image Captioning Model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    
    main(args.config)
