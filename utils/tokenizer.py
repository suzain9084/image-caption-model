import numpy as np
from collections import Counter
import tensorflow as tf

class Vocabulary:
    def __init__(self, min_freq=8, oov_token="<OOV>", max_len=20):
        self.min_freq = min_freq
        self.oov_token = oov_token
        self.max_len = max_len
        self.tokenizer = None

    def build_vocab(self, annotations):
        captions = [ann["caption"].lower() for ann in annotations]

        word_counts = Counter()
        for caption in captions:
            for word in caption.split():
                word_counts[word] += 1

        vocab = {w for w, c in word_counts.items() if c >= self.min_freq}

        processed_captions = [
            "<SOS> " +
            " ".join([w for w in caption.split() if w in vocab]) +
            " <EOS>"
            for caption in captions
        ]

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=self.oov_token)
        self.tokenizer.fit_on_texts(processed_captions)

        return processed_captions

    def texts_to_padded_sequences(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = tf.keras.utils.pad_sequences(sequences, maxlen=self.max_len, padding="post")
        return padded

    def sequence_to_text(self, seq):
        return " ".join([self.tokenizer.index_word.get(idx, self.oov_token) for idx in seq if idx != 0])

    def vocab_size(self):
        return len(self.tokenizer.word_index) + 1

    def load_pretrained_embeddings(self, embedding_path, embedding_dim):
        """Load pretrained GloVe embeddings"""
        embeddings_index = {}
        with open(embedding_path, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = vector

        vocab_size = self.vocab_size()
        embedding_matrix = np.random.normal(
            scale=0.6, size=(vocab_size, embedding_dim)
        )

        for word, idx in self.tokenizer.word_index.items():
            if word in embeddings_index:
                embedding_matrix[idx] = embeddings_index[word]

        self.embedding_matrix = embedding_matrix
        self.embedding_dim = embedding_dim
        return embedding_matrix