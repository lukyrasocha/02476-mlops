from mlops_02476.data import load_dataset
import os.path
import pytest

file_path = "data"


@pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
def test_data():
    dataset = load_dataset()

    assert len(dataset[0].dataset) == 5000
