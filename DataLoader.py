from __future__ import print_function
from tensorflow.keras.datasets import mnist
import numpy as np
import torch


def reshapeImage(images):
    images = images.view(-1, 28 * 28)
    return images


def loadDataset():
    input_shape = (28, 28, 1)
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x = train_x.reshape(train_x.shape[0], -1)
    test_x = test_x.reshape(test_x.shape[0], -1)
    train_x = np.insert(train_x, 0, values=1, axis=1)  # add bias term
    test_x = np.insert(test_x, 0, values=1, axis=1)  # add bias term
    train_x = train_x.astype('float32') / 255.0
    test_x = test_x.astype('float32') / 255.0

    train_x, train_y = filter(train_x, train_y)
    test_x, test_y = filter(test_x, test_y)

    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    test_x = torch.from_numpy(test_x)
    test_y = torch.from_numpy(test_y)

    return train_x.float(), train_y.int(), test_x.float(), test_y.int()


def filter(x, y):
    keep = ((y < 3)) | ((y >= 3) & (y < 6))
    x, y = x[keep], y[keep]
    y = (y < 3)
    y = np.where(y == 1, 1, -1)
    return x, y
