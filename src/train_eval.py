# src/train_eval.py
import os, math, json, random, numpy as np, pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report

from .dataset import load_amazon_dataset
from .absa_extractor import ABSAExtractor
from .features import compute_aspects, reviewer_features, build_meta_matrix
from .model import FusionClassifier

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class RevDS(Dataset):
    def __init__(self, texts, metas, labels, tokenizer, max_len=160):
        self.texts = texts
        self.metas = metas
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['meta'] = torch.tensor(self.metas[i], dtype=torch.float)
        item['labels'] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item

def train_model(train_df, test_df, out_dir='artifacts', cfg=None):
    os.makedirs(out_dir, exist_ok=True)
    cfg = cfg or {}
    model_name = cfg.get('model_name', 'bert-base-uncased')
    max_len = cfg.get('max_len', 160)
    batch = cfg.get('batch', 16)
    epochs = cfg.get('epochs', 3)
    lr = cfg.get('lr', 2e-5)
    warmup_ratio = cfg.get('warmup_ratio', 0.1)

    # ABSA extractor (try PyABSA; else fallback)
    absa = ABSAExtractor()

    # aspects
    train_aspects, _ = compute_aspects(train_df, absa)
    test_aspects, _ = compute_aspects(test_df, absa)

    train_meta_df = reviewer_features(train_df)
    test_meta_df = reviewer_features(test_df)

    X_train_meta, scaler, meta_cols = build_meta_matrix(train_aspects, train_meta_df)
    # apply same scaler to test
    combined_test = pd.concat([pd.DataFrame(test_aspects, columns=[f'asp_{i}' for i in range(test_aspects.shape[1])]), test_meta_df.reset_index(drop=True)], axis=1)
    X_test_meta = scaler.transform(combined_test.values)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_texts = train_df['review_text'].astype(str).tolist()
    test_texts = test_df['review_text'].astype(str).tolist()
    y_train = train_df['label'].astype(int).values
    y_test = test_df['label'].astype(int).values

    ds_tr = RevDS(train_texts, X_train_meta, y_train, tokenizer, max_len)
    ds_va = RevDS(test_texts, X_test_meta, y_test, tokenizer, max_len)

    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False)

    model = FusionClassifier(model_name, meta_dim=X_train_meta.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(dl_tr)
    warmup_steps = int(warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    def run_epoch(loader, train_mode=True):
        if train_mode:
            model.train()
        else:
            model.eval()
        losses = []
        all_preds = []
        all_labels = []
        for batchd in loader:
            input_ids = batchd['input_ids'].to(device)
            attention_mask = batchd['attention_mask'].to(device)
            token_type_ids = batchd.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            meta = batchd['meta'].to(device)
            labels_t = batchd['labels'].to(device)

            with torch.set_grad_enabled(train_mode):
                out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, meta=meta, labels=labels_t)
                loss = out['loss']
                logits = out['logits']
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
            losses.append(loss.item())
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(list(preds))
            all_labels.extend(list(labels_t.detach().cpu().numpy()))
        acc = accuracy_score(all_labels, all_preds)
        p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=1)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except Exception:
            auc = float('nan')
        return np.mean(losses), acc, p, r, f1, auc

    best_f1 = -1
    history = []
    for epoch in range(1, epochs+1):
        tr_loss, tr_acc, tr_p, tr_r, tr_f1, tr_auc = run_epoch(dl_tr, True)
        vl_loss, vl_acc, vl_p, vl_r, vl_f1, vl_auc = run_epoch(dl_va, False)
        history.append({
            'epoch': epoch,
            'train': dict(loss=tr_loss, acc=tr_acc, p=tr_p, r=tr_r, f1=tr_f1, auc=tr_auc),
            'valid': dict(loss=vl_loss, acc=vl_acc, p=vl_p, r=vl_r, f1=vl_f1, auc=vl_auc),
        })
        print(f"Epoch {epoch}: Train F1 {tr_f1:.3f} | Valid F1 {vl_f1:.3f}")
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            ckpt = {
                'model_state': model.state_dict(),
                'tokenizer': model_name,
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
                'meta_cols': meta_cols
            }
            torch.save(ckpt, os.path.join(out_dir, 'best_model.pt'))
            with open(os.path.join(out_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)
            print("Saved best model.")

    # final eval report
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batchd in dl_va:
            input_ids = batchd['input_ids'].to(device)
            attention_mask = batchd['attention_mask'].to(device)
            token_type_ids = batchd.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            meta = batchd['meta'].to(device)
            labels_t = batchd['labels'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, meta=meta)['logits']
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy().tolist())
            val_labels.extend(labels_t.cpu().numpy().tolist())

    print("\nValidation Classification Report:\n")
    print(classification_report(val_labels, val_preds, target_names=['Genuine', 'Fake']))

def predict_texts(model_path, texts, users=None, products=None, ratings=None, times=None, verified=None):
    # load checkpoint
    ck = torch.load(model_path, map_location='cpu')
    model_name = ck.get('tokenizer', 'bert-base-uncased')
    meta_cols = ck['meta_cols']
    mean = np.array(ck['scaler_mean'])
    scale = np.array(ck['scaler_scale'])

    from .absa_extractor import ABSAExtractor
    from .features import reviewer_features
    from transformers import AutoTokenizer
    absa = ABSAExtractor()
    toks = AutoTokenizer.from_pretrained(model_name)

    rows = []
    for i, t in enumerate(texts):
        rows.append({
            'review_text': t,
            'user_id': users[i] if users else f'u{i}',
            'product_id': products[i] if products else f'p{i}',
            'rating': ratings[i] if ratings else 5,
            'review_time': times[i] if times else '2025-01-01',
            'verified_purchase': verified[i] if verified else 1
        })
    df = pd.DataFrame(rows)
    asp = np.vstack([absa.encode(t)[0] for t in df['review_text']])
    mdf = reviewer_features(df)
    combined_test = pd.concat([pd.DataFrame(asp, columns=[f'asp_{i}' for i in range(asp.shape[1])]), mdf.reset_index(drop=True)], axis=1)
    meta_np = (combined_test[meta_cols].values - mean) / scale

    # rebuild model
    from .model import FusionClassifier
    model = FusionClassifier(model_name, meta_dim=meta_np.shape[1])
    model.load_state_dict(ck['model_state'])
    model.eval()

    enc = toks(list(df['review_text'].values), padding=True, truncation=True, max_length=160, return_tensors='pt')
    logits = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], token_type_ids=enc.get('token_type_ids'), meta=torch.tensor(meta_np, dtype=torch.float))['logits']
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)
    return preds, probs
