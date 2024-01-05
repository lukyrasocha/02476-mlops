from torch.optim import Adam
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import nn


class FNN(nn.Module):
  """My awesome model."""

  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(784, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    """Forward pass of the model."""
    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    x = nn.functional.relu(x)
    x = self.fc2(x)
    x = nn.functional.relu(x)
    x = self.fc3(x)
    x = nn.functional.log_softmax(x, dim=1)
    return x


##############################
# REFACTOR TO LIGHTNING MODULE#
##############################


class LightningFNN(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(784, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    # define forward pass
    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    x = nn.functional.relu(x)
    x = self.fc2(x)
    x = nn.functional.relu(x)
    x = self.fc3(x)
    x = nn.functional.log_softmax(x, dim=1)
    return x

  def training_step(self, batch, batch_idx):
    # training step logic
    x, y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    # validation step logic
    x, y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)
    self.log('val_loss', loss)

  def configure_optimizers(self):
    optimizer = Adam(self.parameters(), lr=1e-3)
    return optimizer
