import click
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from models import FNN
from data import load_dataset


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
  print("Training day and night")

  model = FNN()
  # We're only using the training data here
  train_loader = load_dataset(batch_size=64)

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  loss_fn = torch.nn.CrossEntropyLoss()
  training_loss = []

  for epoch in range(10):  # Number of epochs
    for images, targets in train_loader:
      # Flatten the images
      images = images.view(images.size(0), -1)

      optimizer.zero_grad()
      output = model(images)
      loss = loss_fn(output, targets)
      loss.backward()
      optimizer.step()
      training_loss.append(loss.item())

  # Save model

  time = datetime.now().strftime("%Y%m%d%H%M%S")
  torch.save(model, f"models/model_{time}.pt")

  plt.plot(training_loss)
  plt.title("Training Loss")
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  # Save figure to reports/figures
  plt.savefig(f"reports/figures/loss_{time}.png")


if __name__ == "__main__":
  train()
