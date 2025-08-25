# src/cli.py
import click
from train import train_model

@click.group()
def cli():
    """Fake Review Detection CLI"""
    pass

@cli.command()
@click.option('--train', default='data/train.csv', help='Path to training CSV file')
@click.option('--test', default='data/test.csv', help='Path to testing CSV file')
@click.option('--epochs', default=3, help='Number of training epochs')
@click.option('--batch-size', default=16, help='Batch size for training')
@click.option('--lr', default=2e-5, help='Learning rate')
def train(train, test, epochs, batch_size, lr):
    """
    Train and evaluate the Fake Review Detection model.
    """
    print(f"Training on {train} and testing on {test}")
    train_model(train, test, epochs=epochs, batch_size=batch_size, lr=lr)

if __name__ == "__main__":
    cli()
