from __future__ import print_function

import os
import time
import torch
from matplotlib import pyplot as plt
from numpy import mean, std
from DataLoader import loadDataset
from submodular import randomized_submodular, neg_logistic_loss
import seaborn as sns

def predict(x, w):
    with torch.no_grad():
        linear_y = torch.mm(x, w)
        pred = torch.sigmoid(linear_y)
        return pred


def compute_accuracy_loss(x, y, w):
    with torch.no_grad():
        pred = predict(x, w)
        loss = -neg_logistic_loss(y, pred)
        pred_label = torch.where(pred >= 0.5, torch.tensor(1), torch.tensor(-1))
        y = y.view(-1, 1)
        accuracy = (pred_label == y).sum().float() / y.numel()
        # print("Accuracy: ", accuracy.item())
        return accuracy, loss / y.numel()


if __name__ == '__main__':
    #os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # load training data
    train_x, train_y, test_x, test_y = loadDataset()
    seed_ls = torch.arange(0, 3, step=1)
    train_acc_ls = []
    train_loss_ls = []
    test_acc_ls = []
    test_loss_ls = []
    time1 = time.time()
    p = 1
    for seed in seed_ls:
        weights = randomized_submodular(train_x, train_y, p, seed)
        time2 = time.time()
        # print("Training time: ", time2 - time1, "seconds")
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

    # plot
    weights = weights.numpy()[1:785]  # remove bias
    fig, ax = plt.subplots(1, 1)
    plt.title("binary weights")
    sns.heatmap(weights.reshape([28, 28]), annot=False)
    plt.show()
    fig.savefig('binary_weight.png')
    plt.close(fig)