import torch
from mlops_02476 import FNN
import pytest


def test_prediction():
    # Model takes in 784 features and outputs 10 classes
    model = torch.load("models/model_20240105160552.pt")

    # Create random input
    x = torch.rand(1, 1, 28, 28)

    # Make prediction
    prediction = model(x)

    # Check the shape of the prediction
    assert prediction.shape == (1, 10), "Expected prediction to have shape [1, 10]"


def test_model_on_wrong_shape():
    with pytest.raises(ValueError):
        model = FNN()
        x = torch.rand(1, 1, 28, 29)
        prediction = model(x)


@pytest.mark.parametrize("x", [torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28)])
def test_predictions(x):
    model = torch.load("models/model_20240105160552.pt")
    prediction = model(x)
    assert prediction.shape == (1, 10), "Expected prediction to have shape [1, 10]"
