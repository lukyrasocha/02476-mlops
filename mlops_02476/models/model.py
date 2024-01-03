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
