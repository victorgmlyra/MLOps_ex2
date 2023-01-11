import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import transforms

from model import MyAwesomeModel
from load_data import load_data
import hydra

@hydra.main(config_name="basic.yaml")
@click.command()
@click.argument("pretrained_model", type=click.Path(exists=True))
@click.argument("test_imgs_path", type=click.Path())
def main(cfg, pretrained_model, test_imgs_path):
    model = MyAwesomeModel()
    state_dict = torch.load(pretrained_model)
    model.load_state_dict(state_dict)
    test_set = load_data(test_imgs_path)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    with torch.no_grad():
        model.eval()
        acc = []
        for images, labels in testloader:
            # Get the class probabilities
            ps = torch.exp(model(images.float()))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            acc.append(torch.mean(equals.type(torch.FloatTensor)))
        accuracy = sum(acc) / len(acc)
        print(f"Accuracy: {accuracy.item()*100}%")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
