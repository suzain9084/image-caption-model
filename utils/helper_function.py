import torch
import matplotlib.pyplot as plt
import torch

def generate_caption(decoder, features, word_to_idx, idx_to_word, max_len=20, device="cpu"):
    decoder.eval()
    caption = []

    input_word_idx = torch.tensor([[word_to_idx["<SOS>"]]], device=device)
    
    with torch.no_grad():
        for _ in range(max_len):
            word_embedding = decoder.dropout(decoder.linear.weight[input_word_idx])  

            lstm_input = torch.cat((features, word_embedding), dim=1)
            
            lstm_out, _ = decoder.lstm(lstm_input)
            
            output = decoder.linear(lstm_out[:, -1, :])
            predicted_idx = output.argmax(1).item()
            word = idx_to_word[predicted_idx]

            if word == "<EOS>":
                break
            caption.append(word)
            
            input_word_idx = torch.tensor([[predicted_idx]], device=device)

    return " ".join(caption)

def _compute_token_accuracy(logits, targets, ignore_index=0):
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=1)
        valid_mask = targets != ignore_index
        if valid_mask.sum().item() == 0:
            return 0.0
        correct = (predictions[valid_mask] == targets[valid_mask]).sum().item()
        total = valid_mask.sum().item()
        return correct / max(total, 1)
    

def plotAccuracyGraph(metrics, train_losses, val_losses, plots_dir, train_accuracies, val_accuracies):
    try:
        plt.figure(figsize=(8, 5))
        plt.plot(metrics['epochs'], train_losses, label='Train Loss')
        plt.plot(metrics['epochs'], val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'loss.png', dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(metrics['epochs'], train_accuracies, label='Train Accuracy')
        plt.plot(metrics['epochs'], val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Token Accuracy')
        plt.title('Accuracy vs Epochs')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'accuracy.png', dpi=150)
        plt.close()

    except Exception as e:
        print(f"Warning: could not create plots: {e}")