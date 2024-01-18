import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from sklearn.linear_model import LogisticRegression
from torch.autograd import grad, Variable
from models import BinBlock
from models.BinBlock import logistic_loss
from submodular.dynamic_model import *
from submodular.utils import submodularConv, y_times_w, compute_pi, submodularFC


def compute_accuracy(output, train_labels):
    pred = torch.where(torch.sigmoid(output) >= 0.5, torch.tensor(1.), torch.tensor(-1.))
    correct = pred.eq(train_labels.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(train_labels)
    return accuracy


def tempSigmoid(x, temp):
    if x >= 0:
        y = 1.
    else:
        y = 1. / (1. + np.exp(-x / (temp)))
    return y


def printLog(file, *args):
    """ write on both stdout and file"""
    print(*args)
    if file is not None:
        print(*args, file=file)


def update_ri(Wx_r, yw, y, bias):
    ri = compute_pi(Wx=Wx_r, yw=yw) + torch.mul(y.unsqueeze(dim=1), bias)
    return ri


def update_w(new_model, x, targets):
    with torch.enable_grad():
        x = Variable(x, requires_grad=True)
        phi, pred = new_model.forward_res(x)
        # weights of the second layer (w) is
        w = grad(outputs=pred, inputs=phi, grad_outputs=torch.ones_like(pred))[0]
    # compute bias term
    wx = torch.mul(w, phi.detach())
    yw = y_times_w(y=targets, w=w)
    if len(w.shape) == 4:
        wx_sum = torch.unsqueeze(wx.sum(dim=[1, 2, 3]), 1)
    else:
        wx_sum = torch.unsqueeze(wx.sum(dim=1), 1)
    bias = torch.sub(pred.detach(), wx_sum)
    return yw, w, bias


def calculate_fan_in(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    return fan_in


class BinarySubOpt():
    def __init__(self, model, device=torch.device("cpu"), epochs=20, arch="FC2", seed=0, temp=0.05,
                 loss_fc=logistic_loss):
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
        self.use_cuda = self.device == torch.device("cuda")
        self.model = model.to(self.device)
        self.epochs = epochs
        self.save_path = 'result/Submodel/'
        os.makedirs(self.save_path, exist_ok=True)
        self.save_model_path = self.save_path + "{}_init{}_{}.pt".format(self.model._get_name(), self.model.init,
                                                                         self.seed)
        self.tmp_loss = 100.
        self.tmp_acc = 0.
        self.best_loss = 100.
        self.best_acc = 0.
        return

    def submodularOpt(self, trainset, testset):
        self.binarizeConvParams(trainset, testset)
        return self.model

    def binarizeConvParams(self, trainset, testset):
        time1 = time.time()
        result = open('{}{}_init{}_{}.log'.format(self.save_path, self.model._get_name(), self.model.init, self.seed),
                      mode='a', encoding='utf-8')
        output_result = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
        printLog(result, 'Pretrained model:')
        inputs, y = trainset.tensors[0].to(self.device), trainset.tensors[1].to(self.device)
        self.tmp_loss, self.tmp_acc, _, _ = self.save_print_result(trainset, testset, output_result, file=result)
        for epoch in range(self.epochs):
            for idx, (name, m) in reversed(list(enumerate(self.model.body.named_children()))):
                if idx in self.bin_index:
                    accept_freq = 0
                    printLog(result, 'Epoch: {}, Module Name:{}, init = {}'.format(epoch, name, self.model.init))
                    new_model = dynamic_model(layer_idx=idx, model=self.model)
                    x = new_model.forward(inputs)
                    W = self.target_modules[self.bin_index.index(idx)].detach()
                    s = W[0].shape
                    yw, w, bias = update_w(new_model, x, y)
                    row_list = list(range(W.shape[0]))
                    for iteration in range(W.shape[0]):
                        # update the weights for every output channel
                        np.random.seed(self.seed)
                        i = np.random.choice(row_list)
                        row_list.remove(i)
                        W.requires_grad = False
                        beta = np.sqrt(2 / calculate_fan_in(W))
                        if m.Linear == False:
                            # Wx_r
                            W_flatten = W.view(W.shape[0], -1)
                            W_flatten[i][:] = torch.zeros(W_flatten.shape[1], dtype=torch.float32)
                            W.copy_(W_flatten.view(W.shape))
                            Wx_r = m.conv(x).detach()
                            self.restore()
                            ri = update_ri(Wx_r, yw, y, bias)
                            del Wx_r
                            ywi = torch.transpose(yw, 0, 1)[i].unsqueeze(dim=1)
                            Wi = submodularConv(x, m, beta, ywi, W[i], ri)
                            W[i] = Wi.view(s)
                        elif m.Linear == True:
                            # Wx_r
                            W[i, :] = torch.zeros(W.shape[1], dtype=torch.float32).to(self.device)
                            if len(x.shape) != 2:
                                x = x.flatten(1)
                            Wx_r = m.linear(x).detach()
                            self.restore()
                            ri = update_ri(Wx_r, yw, y, bias)
                            del Wx_r
                            ywi = yw[:, i].unsqueeze(dim=1)
                            # update Wi
                            Wi = submodularFC(x, ywi, ri, beta)
                            W[i] = Wi.reshape(1, -1)
                        # check whether to save the change via Accept and Reject algorithm
                        train_loss, train_acc, c2 = self.test_perf(trainset)
                        diff = (self.tmp_loss - train_loss) / self.tmp_loss
                        marg_prob = tempSigmoid(diff, temp=self.temp)
                        new_seed = self.seed + i
                        rand = self.accept_and_reject(new_seed)
                        if rand <= marg_prob:
                            accept_freq += 1
                            self.tmp_loss = train_loss
                            self.tmp_acc = train_acc
                            self.save_params()
                            yw, w, bias = update_w(new_model, x, y)
                            self.target_modules[self.bin_index.index(idx)].data.copy_(W)
                            if m.Linear == False:
                                printLog(result,
                                         'Output channel {}, Train Loss: {}, Train Accuracy: {:.2f}%'.format(i + 1,
                                                                                                             train_loss,
                                                                                                             train_acc))
                            elif m.Linear == True:
                                printLog(result,
                                         'Row {}, Train Loss: {}, Train Accuracy: {:.2f}%'.format(i + 1, train_loss,
                                                                                                  train_acc))

                        else:
                            self.restore()
                    self.updateLastClassifier(inputs, y)
                    train_loss, train_acc, c2 = self.test_perf(trainset)
                    self.tmp_loss = train_loss
                    self.tmp_acc = train_acc
                    printLog(result,
                             'Epoch: {}, Module Name:{}, Accept ratio: {}/{}\n'.format(epoch, name, accept_freq,
                                                                                         W.shape[0]))
                else:
                    pass
            self.save_state()
            self.save_print_result(trainset, testset, output_result, file=result)
        printLog(result, "\nbest testing loss is {} with accuracy {}".format(self.best_loss, self.best_acc))
        time2 = time.time()
        printLog(result, "Training time: ", time2 - time1, "seconds")

    def accept_and_reject(self, seed):
        # sample based on marginal probability
        np.random.seed(seed)  # set seed
        rand = np.random.rand()
        return rand

    def save_print_result(self, trainset, testset, output_result, file):
        train_loss, train_acc, c2 = self.test_perf(trainset)
        test_loss, test_acc, c1 = self.test_perf(testset)
        torch.cuda.empty_cache()
        for key in output_result:
            exec('output_result[key].append(' + key + ')')
        printLog(file,
                 'Train Loss: {}, Train Accuracy: {}/{} ({:.2f}%)'.format(train_loss, c2, len(trainset), train_acc))
        printLog(file, 'Test Loss: {}, Test Accuracy: {}/{} ({:.2f}%)'.format(test_loss, c1, len(testset), test_acc))
        if self.best_loss >= test_loss:
            self.best_loss = test_loss
            self.best_acc = test_acc
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
            loss = self.loss_fc(output_final, test_labels.reshape(-1, 1)).item() / len(test_labels)
            torch.cuda.empty_cache()
        return loss, accuracy, correct
