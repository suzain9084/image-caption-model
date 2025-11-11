from torch.utils.data import Dataset
from utils.tokenizer import Vocabulary
from PIL import Image
from os.path import join
import json

class CaptionDataset(Dataset):
    def __init__(self, annotation_path, image_path, transform, max_len):
        super().__init__()
        print(F"Annotation Path: {annotation_path}")
        self.vocabulary = Vocabulary(min_freq=8, oov_token='<OOV>', max_len=max_len)
        with open(annotation_path, "r") as f:
            self.annotation = json.load(f)

        self.vocabulary.build_vocab(annotations=self.annotation['annotations'])

        self.image_path = image_path
        self.transform = transform
        self.imgId2fileName = {img['id']: img['file_name'] for img in self.annotation['images']}

    def __len__(self):
        return  min(2, len(self.annotation['annotations']))

    def __getitem__(self, index):
        ann = self.annotation['annotations'][index]
        caption = ann['caption']
        padded_sequence = self.vocabulary.texts_to_padded_sequences(caption)

        img_path = join(self.image_path, self.imgId2fileName[ann['image_id']])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, padded_sequence
