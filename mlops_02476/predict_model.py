import torch
import click
from data import load_dataset


@click.command()
@click.argument("model_checkpoint")
@click.argument("data_name")
def predict(
    model_checkpoint,
    data_name,
) -> torch.tensor:

  print(
      f"Loading model from {model_checkpoint} and making a prediction on data set {data_name}")

  model = torch.load(f"{model_checkpoint}")
  dataloader = load_dataset(batch_size=64, setname=data_name)

  predictions = torch.cat([model(batch) for batch, targets in dataloader], 0)

  print(predictions.shape)

  return predictions


if __name__ == "__main__":
  predict()
