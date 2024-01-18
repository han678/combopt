import argparse

import models
from greedy.binaryGreedy import BinaryGreedy
from load_data import load_data, make_dataset, set_seed
from submodular.binaryOpt import *

parser = argparse.ArgumentParser(description='PyTorch Training for MNIST')
parser.add_argument('--data', '-d', default='cifar10', help='dataset: choose from mnist or cifar10')
parser.add_argument('--pretrain-path', '-p', metavar='PATH', default='result/FPmodel/')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
parser.add_argument('--path', type=str, default='~/data', help='Path to dataset')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--weights-init', type=int, default=1, help='choose initialization for weights')
parser.add_argument('--pretrain', type=bool, default=False)
parser.add_argument('--arch', type=str, default='LeNet5', help='model architecture: choose from CNN6 or LeNet5 or FCN2')
parser.add_argument('--opt_strategy', type=str, default='greedy', help='optimization: submodular or greedy')

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # load data
    train_loader, test_loader = load_data(dataset=args.data)
    trainset = make_dataset(train_loader)
    testset = make_dataset(test_loader)
    # build model
    input_channels = 3 if args.data.lower() == "cifar10" else 1
    arch = args.arch  # "Lenet5" 'FCN2' "CNN6"
    # binary optimization
    set_seed(args.seed)
    if arch == "CNN6":
        model = models.CNN6(init=args.weights_init, bias=False, input_channels=input_channels)
    elif arch == 'LeNet5':
        model = models.LeNet5(init=args.weights_init, bias=False, input_channels=input_channels)
    elif arch == 'FCN2':
        model = models.FCN2(init=args.weights_init, n_hidden=10, bias=False, input_channels=input_channels)
    else:
        print("This model is not provided by default.")

    if args.pretrain == True:
        state = torch.load(args.pretrain_path + "{}.pt".format(args.arch))
        print("Accuracy of the pretrained binarized model is: {}".format(state['acc']))
        model.load_state_dict(state['state_dict'])

    if args.opt_strategy == 'submodular':
        opt = BinarySubOpt(model, device=device, epochs=args.epochs, seed=args.seed, arch=arch, temp=0.05)
        opt.submodularOpt(trainset, testset)
    elif args.opt_strategy == 'greedy':
        opt = BinaryGreedy(model, device=device, epochs=args.epochs, seed=args.seed, arch=arch, temp=0.05)
        opt.greedyOpt(trainset, testset)
    else:
        print("The optimization strategy is not provided by default.")
