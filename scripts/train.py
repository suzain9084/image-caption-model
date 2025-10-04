import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

from data.processed.data_processed import CaptionDataset
from models.decoder import DecoderRNN
from models.encoder import EncoderCNN
from utils.transform import ImageTransforms

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(checkpoint_path, encoder, decoder, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['train_loss']

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
        
        outputs = self.decoder(features, embedded_captions[:, :-1, :])
        return outputs


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, captions) in enumerate(progress_bar):
        images = images.to(device)
        captions = captions.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images, captions)
        
        outputs = outputs.reshape(-1, outputs.shape[2])
        targets = captions[:, 1:].reshape(-1)
        
        loss = criterion(outputs, targets)
        
        loss.backward()
        
        if config['training']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        
        optimizer.step()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for images, captions in progress_bar:
            images = images.to(device)
            captions = captions.to(device)
            
            outputs = model(images, captions)
            
            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = captions[:, 1:].reshape(-1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


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
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    print("Initializing models...")
    encoder = EncoderCNN()
    decoder = DecoderRNN(
        embed_size=config['model']['embed_size'],
        hidden_size=config['model']['hidden_size'],
        vocab_size=vocab_size,
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
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
    elif config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training'].get('momentum', 0.9),
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")
    
    if config['training']['use_scheduler']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training']['scheduler_factor'],
            patience=config['training']['scheduler_patience']
        )
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config['training']['resume_checkpoint']:
        checkpoint_path = config['training']['checkpoint_path']
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            start_epoch, train_loss = load_checkpoint(checkpoint_path, encoder, decoder, optimizer)
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}, starting from scratch")
    
    print(f"\nStarting training for {config['training']['num_epochs']} epochs...")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch [{epoch+1}/{config['training']['num_epochs']}]")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        print(f"Training Loss: {train_loss:.4f}")
        
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        if config['training']['use_scheduler']:
            scheduler.step(val_loss)
        
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }
            save_checkpoint(
                checkpoint,
                config['training']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch+1}.pth'
            )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }
            save_checkpoint(
                checkpoint,
                config['training']['checkpoint_dir'],
                'best_model.pth'
            )
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")
    
    print("\nTraining completed!")


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
