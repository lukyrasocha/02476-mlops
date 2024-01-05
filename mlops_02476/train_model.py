import click
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from models import FNN, LightningFNN
from data import load_dataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger


import wandb


# @click.command()
# @click.option("--lr", default=1e-3, help="learning rate to use for training")

def train():
  print("Training day and night")

  # wandb.init(project="corrupt_mnist_fnn", entity="lukyrasocha") #project is the name of the project, entity is the name of the team (for collaboration) or user e.g. lukyrasocha
  # config is a dictionary of all the hyperparameters you want to track
  # wandb.init(config={"lr": 1e-3}, project="corrupt_mnist_fnn", entity="lukyrasocha")

  # model = FNN()

  wandb_logger = WandbLogger(project="LIGHTNING_TEST")

  early_stopping = EarlyStopping(
      monitor='val_loss',
      min_delta=0.00,
      patience=3,
      verbose=True
  )

  trainer = Trainer(
      logger=wandb_logger,
      callbacks=[early_stopping],
      max_epochs=100,
      limit_train_batches=0.2
  )

  model = LightningFNN()

  train_loader, val_loader = load_dataset(batch_size=64)

  trainer.fit(model, train_loader, val_loader)

  # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  # optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)

  # loss_fn = torch.nn.CrossEntropyLoss()
  # training_loss = []

  # for epoch in range(10):  # Number of epochs
  #  for images, targets in train_loader:
  #    # Flatten the images
  #    images = images.view(images.size(0), -1)

  #    optimizer.zero_grad()
  #    output = model(images)
  #    loss = loss_fn(output, targets)
  #    loss.backward()
  #    optimizer.step()
  #    training_loss.append(loss.item())

  #    wandb.log({"loss": loss.item()})  # Log the loss

  # Save model

  # time = datetime.now().strftime("%Y%m%d%H%M%S")
  # torch.save(model, f"models/model_{time}.pt")

  # plt.plot(training_loss)
  # plt.title("Training Loss")
  # plt.xlabel("Iterations")
  # plt.ylabel("Loss")
  # Save figure to reports/figures
  # plt.savefig(f"reports/figures/loss_{time}.png")

  # wandb.log({"Training Loss Plot": wandb.Image(plt)})
  # wandb.finish()  # Finish the WandB run


if __name__ == "__main__":
  train()
