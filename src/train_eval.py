# src/train.py
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from dataset import AmazonReviewsDataset
from model import FusionClassifier

def train_model(train_csv, test_csv, epochs=3, batch_size=16, lr=2e-5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = AmazonReviewsDataset(train_csv)
    test_ds = AmazonReviewsDataset(test_csv)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = FusionClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                absa_features=batch["absa_features"].to(device),
                labels=batch["labels"].to(device)
            )
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                absa_features=batch["absa_features"].to(device),
            )
            preds = torch.argmax(outputs["logits"], dim=1).cpu()
            labels = batch["labels"]
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct/total:.4f}")
    return model
