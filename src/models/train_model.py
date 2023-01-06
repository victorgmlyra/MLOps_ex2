# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import os
import numpy as np
from model import MyAwesomeModel
from load_data import load_data
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_output_filepath', type=click.Path())
@click.argument('figs_output_filepath', type=click.Path())
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def main(input_filepath, model_output_filepath, figs_output_filepath, lr):
    """ Train the network with data from input_filepath 
        and save training log to output filepath
    """
    logger = logging.getLogger(__name__)
    logger.info('Train the network from processed data')

    if not os.path.exists(model_output_filepath):
        os.mkdir(model_output_filepath)
    
    model = MyAwesomeModel()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_set = load_data(input_filepath + '/train.npz')
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    epochs = 10
    losses = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            
            optimizer.zero_grad()
            
            log_ps = model(images.float())
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(trainloader)}")
            losses.append(running_loss/len(trainloader))
    
    torch.save(model.state_dict(), model_output_filepath + '/trained_model.pth')
    plt.plot(losses)
    plt.savefig(figs_output_filepath + '/loss.png')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
