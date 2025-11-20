from torch.utils.data import Dataset
from utils.tokenizer import Vocabulary
from transformers import AutoTokenizer
from PIL import Image
from os.path import join
import json


class CaptionDataset(Dataset):
    def __init__(
        self,
        annotation_path,
        image_path,
        transform,
        max_len,
        min_word_freq=8,
        sample_limit=None,
        use_hf_tokenizer=False,
        hf_tokenizer_name="gpt2",
    ):
        super().__init__()
        print(f"Annotation Path: {annotation_path}")
        with open(annotation_path, "r") as f:
            self.annotation = json.load(f)

        self.image_path = image_path
        self.transform = transform
        self.max_len = max_len
        self.sample_limit = sample_limit
        self.use_hf_tokenizer = use_hf_tokenizer

        self.imgId2fileName = {
            img["id"]: img["file_name"] for img in self.annotation["images"]
        }

        if self.use_hf_tokenizer:
            # Reuse a pretrained text tokenizer (GPT-2 by default) so the decoder
            # can inherit language knowledge. GPT-2 lacks a PAD token, so reuse EOS.
            self.hf_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            self.pad_token_id = self.hf_tokenizer.pad_token_id
        else:
            self.vocabulary = Vocabulary(
                min_freq=min_word_freq, oov_token="<OOV>", max_len=max_len
            )
            self.vocabulary.build_vocab(annotations=self.annotation["annotations"])

    def __len__(self):
        total = len(self.annotation["annotations"])
        if self.sample_limit is not None:
            return min(self.sample_limit, total)
        return total

    def __getitem__(self, index):
        ann = self.annotation["annotations"][index]
        caption = ann["caption"].strip().lower()

        img_path = join(self.image_path, self.imgId2fileName[ann["image_id"]])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.use_hf_tokenizer:
            bos = self.hf_tokenizer.bos_token or self.hf_tokenizer.eos_token
            eos = self.hf_tokenizer.eos_token
            caption_text = f"{bos} {caption} {eos}".strip()
            encoded = self.hf_tokenizer(
                caption_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[input_ids == self.pad_token_id] = -100
            return img, input_ids, attention_mask, labels

        caption_tokens = f"<SOS> {caption} <EOS>"
        padded_sequence = self.vocabulary.texts_to_padded_sequences(caption_tokens)
        return img, padded_sequence
