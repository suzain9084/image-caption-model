"""
Caption Generation and Evaluation Metrics
Implements BLEU, METEOR, ROUGE-L, CIDEr, and SPICE metrics
"""
import torch
import numpy as np
from collections import Counter
import json
from typing import List, Dict
import warnings

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not available. Some metrics will be unavailable.")

try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
except:
    pass

# Disable SSL verification warnings for downloads
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def generate_caption_greedy(model, encoder, decoder, image, vocabulary, max_len=20, device="cpu"):
    model.eval()
    encoder.eval()
    decoder.eval()
    
    image = image.unsqueeze(0).to(device) if len(image.shape) == 3 else image.to(device)
    
    with torch.no_grad():
        # Get image features
        features = encoder(image)
        features = features.unsqueeze(1)  # [1, 1, feature_dim]
        
        # Start with SOS token
        word_to_idx = vocabulary.tokenizer.word_index
        
        sos_idx = word_to_idx.get("<SOS>", 1)
        eos_idx = word_to_idx.get("<EOS>", 2)
        
        # Get index -> word mapping (add 1 offset for padding token)
        idx_to_word_with_pad = {}
        for word, idx in word_to_idx.items():
            # keras tokenizer starts from 1, so we need to map correctly
            idx_to_word_with_pad[idx] = word
        
        caption_words = []
        current_idx = sos_idx
        embed_size = model.embed_layer.embedding_dim
        
        for _ in range(max_len):
            # Create input token
            input_token = torch.tensor([[current_idx]], device=device)
            
            # Get embedding
            embedded = model.embed_layer(input_token)  # [1, 1, embed_size]
            embedded = embedded.squeeze(1)  # [1, embed_size]
            
            # Forward through decoder
            h0 = decoder.init_h(features.squeeze(1)).unsqueeze(0).repeat(decoder.num_layers, 1, 1)
            c0 = decoder.init_c(features.squeeze(1)).unsqueeze(0).repeat(decoder.num_layers, 1, 1)
            
            lstm_out, _ = decoder.lstm(embedded.unsqueeze(1), (h0, c0))
            output = decoder.linear(decoder.dropout(lstm_out[:, -1, :]))
            
            predicted_idx = output.argmax(1).item()
            # predicted_idx is 0-based but vocabulary is 1-based
            if predicted_idx > 0 and predicted_idx in idx_to_word_with_pad:
                word = idx_to_word_with_pad[predicted_idx]
            else:
                word = vocabulary.oov_token
            
            if word == "<EOS>" or predicted_idx == eos_idx:
                break
            
            caption_words.append(word)
            current_idx = predicted_idx
        
        caption = " ".join(caption_words)
        return caption


def compute_bleu_score(predicted: str, references: List[str]):
    if not NLTK_AVAILABLE:
        return {'bleu1': 0.0, 'bleu4': 0.0}
    
    try:
        # Tokenize
        pred_tokens = predicted.lower().split()
        refs_tokens = [ref.lower().split() for ref in references]
        
        # BLEU-1
        bleu1_score = sentence_bleu(
            refs_tokens, 
            pred_tokens, 
            weights=(1, 0, 0, 0),
            smoothing_function=SmoothingFunction().method1
        )
        
        # BLEU-4
        bleu4_score = sentence_bleu(
            refs_tokens, 
            pred_tokens, 
            smoothing_function=SmoothingFunction().method1
        )
        
        return {'bleu1': float(bleu1_score), 'bleu4': float(bleu4_score)}
    except:
        return {'bleu1': 0.0, 'bleu4': 0.0}


def compute_meteor_score(predicted: str, reference: str):
    if not NLTK_AVAILABLE:
        return 0.0
    
    try:
        import nltk
        try:
            from nltk.translate.meteor_score import meteor_score
        except:
            return 0.0
        
        # Tokenize
        pred_tokens = nltk.word_tokenize(predicted.lower())
        ref_tokens = nltk.word_tokenize(reference.lower())
        
        score = meteor_score([ref_tokens], pred_tokens)
        return float(score)
    except:
        return 0.0


def compute_rouge_l_score(predicted: str, reference: str):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference.lower(), predicted.lower())
        return float(scores['rougeL'].fmeasure)
    except:
        return 0.0


def compute_cider_score(predicted: str, references: List[str], n=4):
    try:
        pred_tokens = predicted.lower().split()
        refs_tokens = [ref.lower().split() for ref in references]
        
        # Compute TF-IDF for n-grams
        cider_score = 0.0
        
        for n_gram_order in range(1, n + 1):
            # Get n-grams for predicted
            pred_ngrams = get_ngrams(pred_tokens, n_gram_order)
            
            # Get n-grams for all references
            all_ref_ngrams = []
            for ref_tokens in refs_tokens:
                all_ref_ngrams.extend(get_ngrams(ref_tokens, n_gram_order))
            
            # Count frequencies
            ref_counter = Counter(all_ref_ngrams)
            total_ref_count = len(all_ref_ngrams)
            
            # Compute TF-IDF score
            score = 0.0
            for ngram in pred_ngrams:
                if ngram in ref_counter:
                    # TF-IDF weighted by document frequency
                    tfidf = ref_counter[ngram] / max(total_ref_count, 1)
                    score += tfidf
            
            cider_score += score / max(len(pred_ngrams), 1)
        
        cider_score = cider_score / n
        return float(cider_score)
    except:
        return 0.0


def get_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams


def compute_spice_score(predicted: str, reference: str):
    try:
        pred_tokens = set(predicted.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if len(ref_tokens) == 0:
            return 0.0
        
        # F1 score on word overlap
        intersection = pred_tokens & ref_tokens
        precision = len(intersection) / max(len(pred_tokens), 1)
        recall = len(intersection) / max(len(ref_tokens), 1)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return float(f1)
    except:
        return 0.0


def evaluate_caption_metrics(model, encoder, decoder, dataloader, vocabulary, device, max_samples=None, max_len=20):
    model.eval()
    encoder.eval()
    decoder.eval()
    
    all_bleu1 = []
    all_bleu4 = []
    all_meteor = []
    all_rouge_l = []
    all_cider = []
    all_spice = []
    
    print("Computing caption metrics on validation set...")
    
    with torch.no_grad():
        for idx, (images, captions) in enumerate(dataloader):
            if max_samples and idx >= max_samples:
                break
            
            images = images.to(device)
            
            # Generate captions for each image in batch
            for batch_idx in range(images.shape[0]):
                image = images[batch_idx]
                caption_tokens = captions[batch_idx]
                
                # Generate prediction
                try:
                    predicted_caption = generate_caption_greedy(
                        model, encoder, decoder, image, vocabulary, max_len=max_len, device=device
                    )
                except Exception as e:
                    print(f"Error generating caption: {e}")
                    predicted_caption = ""
                
                # Get reference caption from tokenized sequence
                reference_caption = vocabulary.sequence_to_text(caption_tokens.tolist())
                
                # Remove special tokens and clean
                reference_caption = reference_caption.replace("<SOS>", "").replace("<EOS>", "").strip()
                predicted_caption = predicted_caption.strip()
                
                if predicted_caption == "":
                    continue
                
                # Compute all metrics
                bleu_scores = compute_bleu_score(predicted_caption, [reference_caption])
                all_bleu1.append(bleu_scores['bleu1'])
                all_bleu4.append(bleu_scores['bleu4'])
                
                meteor = compute_meteor_score(predicted_caption, reference_caption)
                all_meteor.append(meteor)
                
                rouge_l = compute_rouge_l_score(predicted_caption, reference_caption)
                all_rouge_l.append(rouge_l)
                
                cider = compute_cider_score(predicted_caption, [reference_caption])
                all_cider.append(cider)
                
                spice = compute_spice_score(predicted_caption, reference_caption)
                all_spice.append(spice)
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1} batches...")
    
    # Compute averages
    avg_scores = {
        'bleu1': float(np.mean(all_bleu1)) if all_bleu1 else 0.0,
        'bleu4': float(np.mean(all_bleu4)) if all_bleu4 else 0.0,
        'meteor': float(np.mean(all_meteor)) if all_meteor else 0.0,
        'rouge_l': float(np.mean(all_rouge_l)) if all_rouge_l else 0.0,
        'cider': float(np.mean(all_cider)) if all_cider else 0.0,
        'spice': float(np.mean(all_spice)) if all_spice else 0.0,
    }
    
    return avg_scores

