import torch
if __name__ == '__main__':
  # Get the data and process it
  train_data = torch.load('data/raw/train_images_6.pt')
  # Standardize the data to have mean 0 and variance 1
  train_data = (train_data - train_data.mean()) / train_data.std()

  # Save the processed data
  torch.save(train_data, 'data/processed/train_images_6.pt')

  print('Data processed.')
  print("Mean is: ", train_data.mean())
  print("Standard deviation is: ", train_data.std())
  print("Shape of data is: ", train_data.shape)
