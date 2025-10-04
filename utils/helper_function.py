import torch

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
