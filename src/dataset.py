import pandas as pd
from sklearn.model_selection import train_test_split

def load_amazon_dataset(train_path, test_path):
    col_names = ["label", "title", "reviewText"]

    train_df = pd.read_csv(train_path, names=col_names, header=None)
    test_df = pd.read_csv(test_path, names=col_names, header=None)

    # Convert labels to 0/1 
    train_df['label'] = train_df['label'].astype(int) - 1
    test_df['label'] = test_df['label'].astype(int) - 1

    # Merge title + reviewText as input
    train_df['text'] = train_df['title'].astype(str) + ". " + train_df['reviewText'].astype(str)
    test_df['text'] = test_df['title'].astype(str) + ". " + test_df['reviewText'].astype(str)

    return train_df[['text', 'label']], test_df[['text', 'label']]
