from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from matplotlib.legend_handler import HandlerLine2D
from torch import nn
from torch.autograd import Variable
import models
from load_data import load_data, set_seed


class fp_model(nn.Module):
    def __init__(self, no_cuda=False, epochs=5, batch_size=128, lr_epochs=10, seed=1,
                 lr=0.01, weight_decay=1e-5, arch='Lenet5', evaluate=False, criterion=models.logistic_loss,
                 num_classes=1, init=1, dataset="MNIST"):
        super(fp_model, self).__init__()
        self.arch = arch
        self.batch_size = batch_size
        self.lr_epochs = lr_epochs
        self.epochs = epochs
        self.seed = seed
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion
        self.cuda = not no_cuda and torch.cuda.is_available()
        self.evaluate = evaluate
        self.num_classes = num_classes
        self.dataset = dataset
        self.init = init
        try:
            exec("self.model = models.{}(init = {}, bias = {})".format(self.arch, self.init, False))
        except Exception as e:
            print('ERROR: specified arch is not suppported')
            exit()
        if self.cuda:
            torch.cuda.manual_seed(self.seed)
            self.model.cuda()
        else:
            torch.manual_seed(self.seed)
        self.param_dict = dict(self.model.named_parameters())
        self.params = []
        for key, value in self.param_dict.items():
            self.params += [{'params': [value], 'lr': self.lr,
                             'weight_decay': self.weight_decay,
                             'key': key}]
        self.optimizer = optim.Adam(self.params, lr=self.lr,
                                    weight_decay=self.weight_decay)
        self.best_acc = 0
        self.best_loss = 100
        self.save_path = 'result/FPmodel/'
        self.save_model_path = self.save_path + self.arch + '_fp.pt'
        if self.evaluate == True:
            self.load_model()
        return

    def load_model(self):
        pretrained_model = torch.load(self.save_model_path)
        best_acc = pretrained_model['acc']
        print("The accuracy of the pretrained full precision {} model is: {}".format(self.arch, best_acc))
        self.model.load_state_dict(pretrained_model['state_dict'])

    def forward(self, x):
        output = self.model(x)
        return output

    def fit(self, train_loader, test_loader):
        self.val_loss = []
        self.val_acc = []
        self.train_loss = []
        self.train_acc = []
        # compute acc for initialized model
        loss, acc, _ = self.compute_perf(train_loader)
        self.train_loss.append(loss)
        self.train_acc.append(acc)
        loss, acc, _ = self.compute_perf(test_loader)
        self.val_loss.append(loss)
        self.val_acc.append(acc)
        # start training
        for epoch in range(1, self.epochs + 1):
            print("Binary classification: Iteration {}:".format(epoch))
            self.adjust_learning_rate(epoch)
            loss, acc = self.train(epoch, train_loader)
            self.train_loss.append(loss)
            self.train_acc.append(acc)
            loss, acc = self.validation(test_loader)
            self.val_loss.append(loss)
            self.val_acc.append(acc)
        pretrained_model = torch.load(self.save_model_path)
        self.model.load_state_dict(pretrained_model['state_dict'])
        self.plot_acc_loss()
        return self.model

    def get_params(self):
        return {"params": self.params}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def save_state(self):
        print('==> Saving model ...')
        state = {
            'acc': self.best_acc,
            'state_dict': self.model.state_dict(),
        }
        for key in state['state_dict'].keys():
            if 'module' in key:
                state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
        torch.save(state, self.save_model_path)

    def train(self, epoch, train_loader):
        self.model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
                one = torch.tensor(1., device=torch.device('cuda:0'))
            else:
                one = torch.tensor(1.)
            data, target = Variable(data), Variable(target)
            output = self.model(data)
            loss = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            output = self.model(data)
            pred = torch.where(torch.sigmoid(output) >= 0.5, one, -one)
            # pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)
        train_acc = 100. * float(correct) / len(train_loader.dataset)
        print('Training: Average loss {:.4f}, Accuracy: {}/{} ({:.2f}%), Best Accuracy: {:.2f}'.format(
            train_loss, correct, len(train_loader.dataset), train_acc, self.best_acc))
        return train_loss, train_acc

    def compute_perf(self, val_loader):
        val_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
                one = torch.tensor(1., device=torch.device('cuda:0'))
            else:
                one = torch.tensor(1.)
            data, target = Variable(data), Variable(target)
            output = self.model(data)
            val_loss += self.criterion(output, target).data.item()
            pred = torch.where(torch.sigmoid(output) >= 0.5, one, -one)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        acc = 100. * float(correct) / len(val_loader.dataset)
        val_loss /= len(val_loader.dataset)
        return val_loss, acc, correct

    def validation(self, val_loader):
        self.model.eval()
        val_loss, val_acc, correct = self.compute_perf(val_loader)
        print('Validation: Average loss: {:.4f}, Accuracy, {}/{} ({:.2f}%), Best Accuracy: {:.2f}'.format(
            val_loss, correct, len(val_loader.dataset), val_acc, self.best_acc))
        if (val_loss < self.best_loss):
            self.best_acc = val_acc
            self.best_loss = val_loss
            if not self.evaluate:
                self.save_state()
        return val_loss, val_acc

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
        self.lr = self.lr * (0.1 ** (epoch // self.lr_epochs))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def plot_acc_loss(self):
        # plot the variation of gradient
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.subplot(111)
        plt.title("Adam")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        x = torch.arange(1, len(self.val_loss) + 1, step=1)
        line1, = plt.plot(x, self.val_loss, marker='.', label='Test')
        line2, = plt.plot(x, self.train_loss, marker='.', label='Train')
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=32)})
        plt.show()
        fig.savefig(self.save_path + self.arch + "_Loss_{}epochs_{}.png".format(self.epochs, self.batch_size))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.subplot(111)
        plt.title("Adam")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        x = torch.arange(1, len(self.val_acc) + 1, step=1)
        line1, = plt.plot(x, self.val_acc, marker='.', label='Test')
        line2, = plt.plot(x, self.train_acc, marker='.', label='Train')
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=32)})
        plt.show()
        fig.savefig(self.save_path + self.arch + "_Acc_{}epochs_{}.png".format(self.epochs, self.batch_size))
        plt.close(fig)


if __name__ == '__main__':
    batchsize = 128
    data = "cifar10"
    train_loader, test_loader = load_data(dataset=data, train_batchsize=batchsize)
    # arch_list = ['Lenet5', 'FCN3', 'CNN6']
    arch_list = ['Lenet5']
    set_seed(0)
    for arch in arch_list:
        model = fp_model(dataset=data, epochs=50, arch=arch, evaluate=False, batch_size=batchsize, init=1)
        # arch='Lenet5' 'FCN3' 'CNN6'
        time1 = time.time()
        fitted_model = model.fit(train_loader, test_loader)
        time2 = time.time()
        print("Training time: ", time2 - time1, "seconds")
