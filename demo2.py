import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from numpy import mean, std

from DataLoader import loadDataset
from demo1 import compute_accuracy_loss
from submodular import tenary_submodular

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # load training data
    train_x, train_y, test_x, test_y = loadDataset()
    n_sample, n_feature = train_x.size()
    p = 0.5
    # initialize w2
    index = torch.arange(0, n_feature, step=1)
    oddIndex = torch.fmod(index, 2)
    w2 = torch.where(oddIndex == 1, .0, .0).reshape(-1, 1)
    n_iters = 1
    seed_ls = torch.arange(0, 1, step=1)
    train_acc_ls = []
    train_loss_ls = []
    test_acc_ls = []
    test_loss_ls = []
    for seed in seed_ls:
        for i in range(n_iters):
            # print("iteration: {}".format(i))
            r = torch.zeros_like(train_y.reshape(-1, 1))
            w1, w2 = tenary_submodular(w2, r, train_x, train_y, p, seed)
        weights = w1 - w2
        # training accuracy
        train_acc, train_loss = compute_accuracy_loss(train_x, train_y, weights)
        train_acc_ls.append(train_acc);
        train_loss_ls.append(train_loss)
        # testing accuracy
        test_accuracy, test_loss = compute_accuracy_loss(test_x, test_y, weights)
        test_acc_ls.append(test_accuracy);
        test_loss_ls.append(test_loss)

    # summarize the result
    print("Training accuracy: {}+-{}, loss: {}+-{}".format(mean(train_acc_ls), std(train_acc_ls),
                                                           mean(train_loss_ls), std(train_loss_ls)))
    print("Testing accuracy: {}+-{}, loss: {}+-{}".format(mean(test_acc_ls), std(test_acc_ls),
                                                          mean(test_loss_ls), std(test_loss_ls)))
    # plots
    weights = weights.numpy()[1:785]  # remove bias
    fig, ax = plt.subplots(1, 1)
    plt.title("ternary weights")
    sns.heatmap(weights.reshape([28, 28]), annot=False)
    plt.show()
    fig.savefig('ternary_weight.png')
    plt.close(fig)
