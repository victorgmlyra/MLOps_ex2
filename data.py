import torch
import numpy as np
import matplotlib.pyplot as plt

def mnist():
    # exchange with the corrupted mnist dataset
    train_dict = np.load('data/corruptmnist/train_0.npz')
    train = list(zip(torch.from_numpy(train_dict['images']), torch.from_numpy(train_dict['labels'])))
    train_dict = np.load('data/corruptmnist/train_1.npz')
    train += list(zip(torch.from_numpy(train_dict['images']), torch.from_numpy(train_dict['labels'])))
    train_dict = np.load('data/corruptmnist/train_2.npz')
    train += list(zip(torch.from_numpy(train_dict['images']), torch.from_numpy(train_dict['labels'])))
    train_dict = np.load('data/corruptmnist/train_3.npz')
    train += list(zip(torch.from_numpy(train_dict['images']), torch.from_numpy(train_dict['labels'])))
    train_dict = np.load('data/corruptmnist/train_4.npz')
    train += list(zip(torch.from_numpy(train_dict['images']), torch.from_numpy(train_dict['labels'])))
    test_dict = np.load('data/corruptmnist/test.npz')
    test = list(zip(test_dict['images'], test_dict['labels']))
    return train, test


if __name__ == "__main__":
    tr, te = mnist()

    i = 8
    print(tr[i][0].shape)
    plt.imshow(tr[i][0])
    plt.show()