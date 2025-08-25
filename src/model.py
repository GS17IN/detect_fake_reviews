# src/model.py
import torch
import torch.nn as nn
from transformers import BertModel

class FusionClassifier(nn.Module):
    def __init__(self, absa_dim=5, num_labels=2):
        super(FusionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden_size = self.bert.config.hidden_size

        # Fusion: BERT [CLS] + ABSA features
        self.fc = nn.Linear(hidden_size + absa_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, num_labels)

    def forward(self, input_ids, attention_mask, absa_features, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token

        fused = torch.cat([cls_emb, absa_features], dim=1)
        x = self.dropout(torch.relu(self.fc(fused)))
        logits = self.out(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
