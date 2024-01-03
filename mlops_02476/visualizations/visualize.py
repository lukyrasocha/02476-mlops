import click
import torch
import numpy as np

from sklearn.manifold import TSNE
# or from mlops_02476.data import load_dataset
from mlops_02476 import load_dataset
from matplotlib import pyplot as plt


class ModifiedFNN(torch.nn.Module):
  def __init__(self, original_model):
    super().__init__()
    self.fc1 = original_model.fc1
    self.fc2 = original_model.fc2

  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    x = torch.nn.functional.relu(x)
    x = self.fc2(x)
    x = torch.nn.functional.relu(x)
    return x


@click.command()
@click.argument("model_checkpoint")
@click.argument("data_name")
def visualize_features_before_last_layer(model_checkpoint, data_name):

  print("Visualizing features before last layer")
  trained_model = torch.load(f"{model_checkpoint}")
  trained_model.eval()

  modified_model = ModifiedFNN(trained_model)

  dataloader = load_dataset(batch_size=64, setname=data_name)

  # Extract features
  features = []
  labels = []
  for inputs, label in dataloader:
    with torch.no_grad():
      feature = modified_model(inputs)
    features.extend(feature.cpu().numpy())
    labels.extend(label.cpu().numpy())

  # Dimensionality reduction using t-SNE
  features = np.array(features)
  tsne = TSNE(n_components=2, random_state=0)
  features_2d = tsne.fit_transform(features)

  # Visualization
  plt.figure(figsize=(10, 10))
  for label in set(labels):
    indices = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=label)
  plt.legend()
  plt.title("Feature Visualization")

  # Save the plot
  model_time = model_checkpoint.split("_")[-1].split(".")[0]
  plt.savefig(
      f'reports/figures/feature_visualization_MODEL_{model_time}_DATA_{data_name}.png')


if __name__ == '__main__':
  visualize_features_before_last_layer()
