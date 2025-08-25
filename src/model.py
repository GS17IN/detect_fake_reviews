
# src/model.py
import torch
from torch import nn
from transformers import AutoModel

class FusionClassifier(nn.Module):
    """Fusion model that concatenates BERT CLS embedding with metadata/aspect MLP.

    Architecture:
      - BERT encoder (AutoModel)
      - meta_mlp: projects metadata vector to a smaller hidden vector
      - classifier: MLP over [CLS || meta_h] -> logits (2 classes)
    """
    def __init__(self, model_name: str, meta_dim: int, hidden_meta: int = 64, dropout: float = 0.2):
        super().__init__()
        # load pretrained encoder
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # small MLP for metadata/aspect features
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden_meta),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # classification head over fused representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + hidden_meta, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, meta=None, labels=None):
        # BERT forward
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # CLS token embedding as sentence representation
        cls = outputs.last_hidden_state[:, 0, :]

        # project metadata
        meta_h = self.meta_mlp(meta)

        # fuse and classify
        fused = torch.cat([cls, meta_h], dim=1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}
