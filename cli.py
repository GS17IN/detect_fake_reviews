import click
from src.dataset import load_amazon_dataset
from src.train_eval import train_model, predict

@click.group()
def cli():
    """Fake Review Detection CLI"""
    pass

@cli.command()
@click.option('--train', default='data/train.csv', help='Path to training CSV')
@click.option('--test', default='data/test.csv', help='Path to test CSV')
def train(train, test):
    """Train model on Amazon dataset"""
    train_df, test_df = load_amazon_dataset(train, test)
    train_model(train_df, test_df)

@cli.command()
@click.option('--text', prompt='Enter a review text')
def infer(text):
    """Run inference on a single review"""
    label = predict(text)
    click.echo(f"Prediction: {'Fake' if label==1 else 'Genuine'}")

if __name__ == "__main__":
    cli()

