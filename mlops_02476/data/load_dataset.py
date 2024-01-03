import torch
from torch.utils.data import TensorDataset, DataLoader


def load_dataset(batch_size=64, setname='0'):
  # Load the data
  train_images = torch.load(f'data/processed/train_images_{setname}.pt')
  # test_images = torch.load('data/processed/test_images.pt')

  train_targets = torch.load(f'data/processed/train_target_{setname}.pt')
  # test_targets = torch.load('data/corruptmnist/test_target.pt')

  # Creating tensor datasets
  train_dataset = TensorDataset(train_images, train_targets)
  # test_dataset = TensorDataset(test_images, test_targets)

  # Creating data loaders
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader  # , test_loader
