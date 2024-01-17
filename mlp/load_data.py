import torch
import torchvision
import numpy as np
import os
import random
import numpy
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms


def load_data(dataset, train_batchsize=None, test_batchsize=None):
    '''
    :param dataset: MNIST or CIFAR10
    :param class_index: construct one versus rest dataset
    :param batchsize: Size of mini-batch.
    :return: Dataloader of pytorch in train mode and PIL image, as well as label in attack mode.
    '''
    channels = 1 if "mnist" in dataset.lower() else 3
    transform = process_input(channels)

    if dataset.lower() == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        trainset = get_binary_class(trainset)
        testset = get_binary_class(testset)
    elif dataset.lower() == "mnist":
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        trainset.targets = torch.where(trainset.targets.clone().detach() < 5, torch.tensor(1), torch.tensor(-1))
        testset.targets = torch.where(testset.targets.clone().detach() < 5, torch.tensor(1), torch.tensor(-1))
    else:
        raise ValueError("Unknown dataset.")

    batchsize1 = len(trainset) if train_batchsize == None else train_batchsize
    batchsize2 = len(testset) if test_batchsize == None else test_batchsize
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize2, shuffle=False)
    return train_loader, test_loader


def get_binary_class(dataset):
    target = np.array(dataset.targets)
    label = torch.from_numpy(target)
    index = (label <= 5).nonzero().reshape(1, -1).tolist()[0]
    dataset.targets = np.where(target < 3, 1, -1).tolist()
    newset = torch.utils.data.Subset(dataset, index)
    return newset


def process_input(channel):
    if channel == 1:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)), # normalize x from (0,1) to (-1,1)
        ])
    elif channel == 3:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(1.4914, 1.4822, 1.4465), std=(0.247, 0.2435, 0.2616)),
        ])
    return transform


def make_dataset(data_loader):
    images, labels = next(iter(data_loader))
    dataset = TensorDataset(images, labels)
    return dataset


def set_seed(seed):
    """Set all seeds to make results reproducible (deterministic mode).
       When seed is None, disables deterministic mode.
    :param seed: an integer to your choosing
    """
    if seed is not None:
        seed = int(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        numpy.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
