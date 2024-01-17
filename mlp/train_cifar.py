import argparse
import models
from greedy.binaryGreedy import BinaryGreedy
from load_data import load_data, make_dataset, set_seed
from submodular.binaryOpt import *

parser = argparse.ArgumentParser(description='PyTorch Training for CIFAR10')
parser.add_argument('--data', '-d', default='cifar10')
parser.add_argument('--pretrain-path', '-p', metavar='PATH', default='result/FPmodel/')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
parser.add_argument('--path', type=str, default='~/data', help='Path to dataset')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--weights-init', type=int, default=1, help='choose initialization for weights')
parser.add_argument('--pretrain', type=bool, default=False)
parser.add_argument('--arch', type=str, default='CNN6', help='model architecture: CNN6 or Lenet5')

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # load data
    train_loader, test_loader = load_data(dataset=args.data)
    trainset = make_dataset(train_loader)
    testset = make_dataset(test_loader)
    intChannel = 3 if args.data.lower() == "cifar10" else 1
    result_path = 'result/CIFAR10/' if args.data.lower() == "cifar10" else 'result/MNIST/'
    os.makedirs(result_path, exist_ok=True)
    # binary submodular optimization
    set_seed(args.seed)
    arch = args.arch
    exec("model = models.{}(init = {}, bias = {}, intChannel={})".format(arch, args.weights_init, False, intChannel))
    binary_sub_opt = BinarySubOpt(model, device=device, epochs=args.epochs, seed=args.seed, arch=arch, temp=0.05, path=result_path)
    if args.pretrain == True:
        binary_sub_opt.load_state()
    binary_sub_opt.submodularOpt(trainset, testset)
    del model, binary_sub_opt

    # binary greedy optimization
    set_seed(args.seed)
    exec("model = models.{}(init = {}, bias = {}, intChannel={})".format(arch, args.weights_init, False,
                                                                         intChannel))
    binary_greedy_opt = BinaryGreedy(model, device=device, epochs=args.epochs, seed=args.seed, arch=arch,
                                     temp=0.05, path=result_path)
    if args.pretrain == True:
        binary_greedy_opt.load_state()
    binary_greedy_opt.greedyOpt(trainset, testset)
    del model, binary_greedy_opt
