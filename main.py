import argparse
import sys

import torch
import click
from torchvision import transforms

from data import mnist
from model import MyAwesomeModel
from torch import nn, optim

import matplotlib.pyplot as plt


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)


    train_set, _ = mnist()
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
    
    torch.save(model.state_dict(), 'trained_model.pth')
    plt.plot(losses)
    plt.show()


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    _, test_set = mnist()
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
        print(f'Accuracy: {accuracy.item()*100}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    