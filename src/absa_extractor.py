# src/absa_extractor.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define aspects relevant for product reviews
ASPECTS = ["quality", "delivery", "price", "packaging", "service"]

class ABSAExtractor:
    def __init__(self, model_name="yangheng/deberta-v3-base-absa-v1.1", device=None):
        """
        Aspect-Based Sentiment Analysis (ABSA) extractor
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extract(self, title: str, review: str):
        #Extract aspect-based sentiment features for a given review.
        #Uses both title + review text together.
        text = f"{title}. {review}"
        features = []

        for aspect in ASPECTS:
            inp = self.tokenizer(
                f"{text} [ASP] {aspect}",
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**inp)
                score = torch.argmax(out.logits, dim=1).item()  # 0=negative,1=neutral,2=positive

            # Normalize scores to [-1, 0, +1] for the classifier
            features.append(score - 1)

        return features
