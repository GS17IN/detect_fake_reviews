# src/dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from absa_extractor import ABSAExtractor

class AmazonReviewsDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name="bert-base-uncased", max_len=256):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.absa = ABSAExtractor()

        # Normalize labels (assumes dataset labels are 1/2 → convert to 0/1)
        if "label" in self.data.columns:
            self.data["label"] = self.data["label"].apply(lambda x: 0 if x == 1 else 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        title = str(row.get("title", ""))
        review = str(row.get("reviewText", ""))

        # BERT encoding
        encodings = self.tokenizer(
            f"{title}. {review}",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        )

        # ABSA features
        absa_features = self.absa.extract(title, review)

        item = {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "absa_features": torch.tensor(absa_features, dtype=torch.float)
        }

        if "label" in row:
            item["labels"] = torch.tensor(row["label"], dtype=torch.long)

        return item
