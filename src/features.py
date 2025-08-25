
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .absa_extractor import ABSAExtractor

def compute_aspects(df, absa: ABSAExtractor):
    # returns np.array (n_examples, n_aspects)
    vecs = []
    raw = []
    for t in df['review_text'].astype(str).tolist():
        v, r = absa.encode(t)
        vecs.append(v)
        raw.append(r)
    return np.vstack(vecs), raw

def reviewer_features(df):
    f = pd.DataFrame()
    f['char_len'] = df['review_text'].astype(str).str.len()
    f['word_len'] = df['review_text'].astype(str).str.split().apply(len)
    f['exclaim'] = df['review_text'].astype(str).str.count('!')
    f['caps_ratio'] = df['review_text'].astype(str).apply(lambda s: sum(1 for c in s if c.isupper()) / max(1, len(s)))
    if 'review_time' in df.columns:
        f['_dt'] = pd.to_datetime(df['review_time'], errors='coerce')
        f['dayofweek'] = f['_dt'].dt.dayofweek.fillna(0).astype(int)
        f['month'] = f['_dt'].dt.month.fillna(0).astype(int)
    else:
        f['dayofweek'] = 0; f['month'] = 0
    if 'user_id' in df.columns:
        f['user_review_count'] = df.groupby('user_id')['review_text'].transform('count')
        f['user_product_diversity'] = df.groupby('user_id')['product_id'].transform('nunique') if 'product_id' in df.columns else 1
    else:
        f['user_review_count'] = 1; f['user_product_diversity'] = 1
    if 'rating' in df.columns and 'product_id' in df.columns:
        prod_mean = df.groupby('product_id')['rating'].transform('mean')
        f['rating_dev'] = (df['rating'] - prod_mean).fillna(0)
    else:
        f['rating_dev'] = 0
    f['verified_flag'] = df.get('verified_purchase', 0).fillna(0)
    return f

def build_meta_matrix(aspect_matrix, meta_df):
    combined = pd.concat([pd.DataFrame(aspect_matrix, columns=[f'asp_{i}' for i in range(aspect_matrix.shape[1])]), meta_df.reset_index(drop=True)], axis=1)
    scaler = StandardScaler().fit(combined.values)
    scaled = scaler.transform(combined.values)
    return scaled, scaler, combined.columns.tolist()
