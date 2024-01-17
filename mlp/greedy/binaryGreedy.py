import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.legend_handler import HandlerLine2D
from sklearn.linear_model import LogisticRegression
from torch.autograd import Variable

from models import BinBlock, logistic_loss
from submodular.binaryOpt import printLog, tempSigmoid, calculate_fan_in
from submodular.dynamic_model import dynamic_model


def greedyConv(x, m, beta, W, i, Wx, targets, new_model):
    with torch.no_grad():
        device = x.device
        W_zero = torch.zeros_like(W).to(device)
        W_init = W_zero.clone()
        Wi_zero = torch.zeros_like(W[i].unsqueeze(dim=0))
        W_init[i] = torch.where(W[i] >= 0, beta, -beta).to(device)
        Wi = W_init[i].view(Wi_zero.shape[0], -1)
        ConvLayer = BinBlock(input_channels=m.input_channels, output_channels=m.output_channels,
                             kernel_size=m.kernel_size,
                             stride=m.stride, padding=m.padding, bias=m.bias,
                             previous_conv=m.previous_conv).conv.to(device)
        ConvLayer.weight.data = W_init
        Wx.add_(ConvLayer(x))
        act = nn.ReLU(inplace=True)
        pred = new_model.forward_net3(act(Wx)).detach()
        loss = logistic_loss(pred, targets)
        del pred
        for j in range(Wi.shape[1]):
            W_movement = W_zero.clone()
            Wi_movement = Wi_zero.view(Wi_zero.shape[0], -1).clone()
            Wi_movement[0][j] = 1
            W_movement[i] = Wi_movement.view(W[i].shape)
            Wi_next = torch.add(Wi, - Wi[0][j] * 2 * Wi_movement)
            ConvLayer.weight.data = W_movement
            increasement = ConvLayer(x).detach()
            Wx_next = torch.add(Wx, (- Wi[0][j].item()) * 2 * increasement)
            pred = new_model.forward_net3(act(Wx_next)).detach()
            loss_next = logistic_loss(pred, targets)
            del pred
            diff = loss_next - loss
            if diff <= 0:  # accept the change of the jth element of Wi
                Wx = Wx_next
                Wi = Wi_next
                loss = loss_next.clone()
        return Wi


def greedy_FC(x, Wx, Wi, beta, targets, new_model, i):
    with torch.no_grad():
        device = x.device
        n_sample, n_feature = x.size()
        # intialize Wi with -beta: n_feature*1 tensor
        Wi = torch.where(Wi >= 0, beta, -beta).to(device)
        Wix = torch.mm(x, Wi.reshape(-1, 1))
        act = nn.ReLU(inplace=True)
        Wx[:, i] = Wix.squeeze(dim=1)
        pred = new_model.forward_net3(act(Wx)).detach()
        loss = logistic_loss(pred, targets)
        del pred
        for j in range(n_feature):
            Wi_next = Wi.clone()
            Wi_next[j] = - Wi[j]
            unit_increase = x[:, j].view(n_sample, -1) * 1
            Wix_next = torch.add(Wix, (Wi_next[j].item()) * 2 * unit_increase)
            Wx[:, i] = Wix_next.squeeze(dim=1)
            pred_next = new_model.forward_net3(act(Wx))
            loss_next = logistic_loss(pred_next, targets)
            del pred_next
            diff = loss_next - loss
            if diff <= 0:
                Wix = Wix_next
                Wi = Wi_next
                loss = loss_next
    return Wi


class BinaryGreedy():

    def __init__(self, model, device=torch.device("cpu"), epochs=20, arch="FC2", seed=0, temp=0.05,
                 loss_fc=logistic_loss, path='result/MNIST/'):
        # count the number of Conv2d and Linear
        count_targets = 0
        self.bin_index = []
        self.arch = arch
        self.loss_fc = loss_fc
        self.seed = seed
        self.temp = temp
        pos = 0
        for m in model.body.children():
            if isinstance(m, BinBlock):
                self.bin_index.append(pos)
                count_targets = count_targets + 1
            pos += 1
        self.saved_params = []
        self.target_modules = []
        self.num_of_params = len(self.bin_index)
        for idx, block in enumerate(model.body.children()):
            if idx in self.bin_index:
                for m in block.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        tmp = m.weight.data.clone()
                        self.saved_params.append(tmp)
                        self.target_modules.append(m.weight)

        self.device = device
        self.model = model.to(self.device)
        self.epochs = epochs
        self.save_path = path + 'Binary/GreedyCD/'
        os.makedirs(self.save_path, exist_ok=True)
        self.save_model_path = self.save_path + "{}_init{}_{}_{}.pt".format(self.model._get_name(), self.model.init,
                                                                            self.seed, self.model.n_hidden)
        self.tmp_loss = 100.
        self.tmp_acc = 0.
        self.best_loss = 100.
        self.best_acc = 0.
        self.accept_ratio_list = []
        self.std = 0
        return

    def load_state(self):
        state = torch.load(self.save_model_path)
        print("Accuracy of the pretrained binarized model is: {}".format(state['acc']))
        self.model.load_state_dict(state['state_dict'])

    def save_state(self):
        print('==> Saving model ...')
        state = {
            'acc': self.tmp_acc,
            'state_dict': self.model.state_dict(),
        }
        for key in state['state_dict'].keys():
            if 'module' in key:
                state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
        torch.save(state, self.save_model_path)

    def greedyOpt(self, trainset, testset):
        self.binarizeConvParams(trainset, testset)
        return self.model

    def binarizeConvParams(self, trainset, testset):
        time1 = time.time()
        result = open('{}{}_init{}_{}_{}.log'.format(self.save_path, self.model._get_name(), self.model.init, self.seed,
                                                     self.model.n_hidden), mode='a', encoding='utf-8')
        output_result = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
        printLog(result, 'Pretrained model:')
        inputs, targets = trainset.tensors[0].to(self.device), trainset.tensors[1].to(self.device)
        self.tmp_loss, self.train_acc, _, _ = self.save_print_result(trainset, testset, output_result, file=result)
        with torch.no_grad():
            for epoch in range(self.epochs):
                for idx, (name, m) in reversed(list(enumerate(self.model.body.named_children()))):
                    if idx in self.bin_index:
                        accept_freq = 0.
                        printLog(result, 'Epoch: {}, Module Name:{}, init = {}'.format(epoch, name, self.model.init))
                        new_model = dynamic_model(layer_idx=idx, model=self.model)
                        x = new_model.forward(inputs)
                        W = self.target_modules[self.bin_index.index(idx)].detach()
                        beta = np.sqrt(2 / calculate_fan_in(W))
                        row_list = list(range(W.shape[0]))
                        if m.Linear == False:
                            s = W[0].shape
                            # update the weights for every output channel
                            for iteration in range(W.shape[0]):
                                np.random.seed(self.seed)
                                i = np.random.choice(row_list)
                                row_list.remove(i)
                                W_flatten = W.view(W.shape[0], -1)
                                W_flatten[i][:] = torch.zeros(W_flatten.shape[1], dtype=torch.float32)
                                W.copy_(W_flatten.view(W.shape))
                                Wx_r = m.conv(x).detach()
                                self.restore()
                                Wi = greedyConv(x, m, beta, W, i, Wx_r, targets, new_model)
                                W[i] = Wi.view(s)
                                train_loss, _, train_acc, c2 = self.test_perf(trainset)
                                diff = (self.tmp_loss - train_loss) / self.tmp_loss
                                marg_prob = tempSigmoid(diff, temp=self.temp)
                                new_seed = self.seed + i
                                rand = self.accept_and_reject(new_seed)
                                if rand <= marg_prob:
                                    accept_freq += 1
                                    self.tmp_loss = train_loss
                                    self.tmp_acc = train_acc
                                    self.save_params()
                                    self.target_modules[self.bin_index.index(idx)].data.copy_(W)
                                    printLog(result,
                                             'Output channel {}, Train Loss: {}, Train Accuracy: {:.2f}%'.format(i + 1,
                                                                                                                 train_loss,
                                                                                                                 train_acc))
                                else:
                                    self.restore()
                        elif m.Linear == True:
                            if len(x.shape) != 2:
                                x = x.flatten(1)
                            for iteration in range(W.shape[0]):
                                np.random.seed(self.seed)
                                i = np.random.choice(row_list)
                                row_list.remove(i)
                                # remove the ith row of W
                                W.requires_grad = False
                                W[i, :] = torch.zeros(W.shape[1], dtype=torch.float32, device=W.device)
                                Wx_r = m.linear(x).detach()
                                self.restore()
                                W[i, :] = greedy_FC(x, Wx_r, W[i, :], beta, targets, new_model, i)
                                # check whether to save the change via Accept and Reject algorithm
                                train_loss, _, train_acc, c2 = self.test_perf(trainset)
                                diff = self.tmp_loss - train_loss
                                marg_prob = tempSigmoid(diff, temp=self.temp)
                                new_seed = self.seed + i
                                rand = self.accept_and_reject(new_seed)
                                if rand <= marg_prob:
                                    accept_freq += 1
                                    self.tmp_loss = train_loss
                                    self.tmp_acc = train_acc
                                    self.save_params()
                                    self.target_modules[self.bin_index.index(idx)].data.copy_(W)
                                    printLog(result, 'Row: {}, Train Loss: {}, Train Accuracy: {:.2f}%'.format(i + 1,
                                                                                                               train_loss,
                                                                                                               train_acc))
                                else:
                                    self.restore()
                        self.updateLastClassifier(inputs, targets)
                        train_loss, train_loss_std, train_acc, c2 = self.test_perf(trainset)
                        self.tmp_loss = train_loss
                        self.tmp_acc = train_acc
                        printLog(result,
                                 '\nEpoch: {}, Module Name:{}, Accept ratio: {}/{}'.format(epoch, name, accept_freq,
                                                                                           W.shape[0]))
                    else:
                        pass
                self.save_print_result(trainset, testset, output_result, file=result)
        printLog(result, "best testing loss is {} with std {} and accuracy {}\n\n".format(self.best_loss, self.std,
                                                                                          self.best_acc))
        time2 = time.time()
        printLog(result, "Training time: ", time2 - time1, "seconds")
        self.plot(output_result)

    def accept_and_reject(self, seed):
        # sample based on marginal probability
        np.random.seed(seed)
        rand = np.random.rand()
        return rand

    def save_print_result(self, trainset, testset, output_result, file):
        train_loss, train_loss_std, train_acc, c2 = self.test_perf(trainset)
        test_loss, test_loss_std, test_acc, c1 = self.test_perf(testset)
        torch.cuda.empty_cache()
        for key in output_result:
            exec('output_result[key].append(' + key + ')')
        printLog(file,
                 'Train Loss: {} with standard deviation {}, Accuracy: {}/{} ({:.2f}%)'.format(train_loss,
                                                                                               train_loss_std, c2,
                                                                                               len(trainset),
                                                                                               train_acc))
        printLog(file,
                 'Test Loss: {} with standard deviation {}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, test_loss_std,
                                                                                              c1, len(testset),
                                                                                              test_acc))
        if self.best_loss >= test_loss:
            self.best_loss = test_loss
            self.best_acc = test_acc
            self.std = test_loss_std
        return train_loss, train_acc, test_loss, test_acc

    def updateLastClassifier(self, train_features, train_labels):
        layer_set = list(self.model.body.named_children())
        last_layer = list(self.model.body.named_children())[-1][0]
        last_layer_idx = len(layer_set) - 1
        with torch.no_grad():
            new_model = dynamic_model(layer_idx=last_layer_idx, model=self.model)
            x0 = Variable(train_features, requires_grad=False)
            x1 = new_model.forward(x0).cpu()
        x2, train_labels = np.array(x1.data.cpu().clone().flatten(1)), np.array(train_labels.cpu())
        lr = LogisticRegression(fit_intercept=True, max_iter=100, solver='sag')
        lr.fit(x2, train_labels)
        W_init = eval("self.model.body." + last_layer + ".weight.detach()")
        W = torch.tensor(lr.coef_[0], dtype=torch.float).to(self.device)  # coef of Logistic regression
        W = W.view(W_init.size())
        W_init.copy_(W)
        # fit the bias term in the last classifier
        b_init = eval("self.model.body." + last_layer + ".bias.detach()")
        b = torch.tensor(lr.intercept_[0], dtype=torch.float).to(self.device)  # intercept of Logistic regression
        b_init.copy_(b)
        self.save_params()

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def save_state(self):
        print('==> Saving model ...')
        state = {
            'acc': self.tmp_acc,
            'state_dict': self.model.state_dict(),
        }
        for key in state['state_dict'].keys():
            if 'module' in key:
                state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
        torch.save(state, self.save_model_path)

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def test_perf(self, testset):
        self.model.eval()
        # For every epoch, compute loss and testing accuracy
        test_features, test_labels = testset.tensors[0].to(self.device), testset.tensors[1].to(self.device)
        with torch.no_grad():
            output_final = self.model(test_features)
            pred = torch.where(torch.sigmoid(output_final) >= 0.5, torch.tensor(1).to(self.device),
                               torch.tensor(-1).to(self.device))
            correct = pred.eq(test_labels.view_as(pred)).sum().item()
            accuracy = 100. * correct / len(test_labels)
            loss_mean, loss_std = self.loss_fc(output_final, test_labels.reshape(-1, 1), returnSTD=True)
        torch.cuda.empty_cache()
        return loss_mean.item(), loss_std.item(), accuracy, correct

    def plot(self, output_result):
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.subplot(111)
        plt.title("Loss / Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        x = np.arange(1, len(output_result['test_loss']) + 1, step=1)
        line1, = plt.plot(x, output_result['test_loss'][0:], marker='.', label='Test')
        line2, = plt.plot(x, output_result['train_loss'][0:], marker='.', label='Train')
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=32)})
        plt.show()
        fig.savefig(
            self.save_path + "{}_Loss_init{}_{}_{}.png".format(self.model._get_name(), self.model.init, self.seed,
                                                               self.model.n_hidden))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.subplot(111)
        plt.title("Accuracy / Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        x = np.arange(1, len(output_result['test_acc']) + 1, step=1)
        line1, = plt.plot(x, output_result['test_acc'][0:], marker='.', label='Test')
        line2, = plt.plot(x, output_result['train_acc'][0:], marker='.', label='Train')
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=32)})
        plt.show()
        fig.savefig(
            self.save_path + "{}_Accuracy_init{}_{}_{}.png".format(self.model._get_name(), self.model.init, self.seed,
                                                                   self.model.n_hidden))
        plt.close(fig)
