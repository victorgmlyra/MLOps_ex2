from tests import _PATH_DATA
from src.models.load_data import load_data
import os.path
import pytest

@pytest.mark.skipif(not os.path.exists(_PATH_DATA + '/train.npz'), reason="Data files not found")
def test_data():
    data_train = load_data(_PATH_DATA + '/train.npz')
    data_test = load_data(_PATH_DATA + '/test.npz')
    assert len(data_train) == 25000, "Train dataset did not have the correct number of samples"
    assert len(data_test) == 5000, "Test dataset did not have the correct number of samples"
    unique_labels = []
    for img, label in data_train:
        unique_labels.append(label)
        assert img.shape == (28,28), 'Input images have wrong shape'
    for img, label in data_test:
        assert img.shape == (28,28), 'Input images have wrong shape'

    for i in range(10):
        assert i in unique_labels, 'Not all labels are represented in the train dataset'