from src.models.model import MyAwesomeModel
import numpy as np
import torch

def test_model():
    model = MyAwesomeModel()
    image = np.ones((1, 28, 28))
    image = torch.from_numpy(image)
    images = image.view(image.shape[0], -1)
    log_ps = model(images.float())
    assert log_ps.shape == (1, 10)