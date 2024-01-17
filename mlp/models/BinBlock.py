import torch
from torch import nn


def logistic_loss(output, target, returnSTD=False):
    target = target.view(-1, 1)
    output = output.view(-1, 1)
    loss = torch.log(1. + torch.exp(-target.mul(output)))
    if returnSTD==False:
        return loss.mean()
    else:
        return loss.mean(), loss.std()

def initialization(init, module):
    if init == 1:
        for m in module.children():
            if isinstance(m, BinBlock):
                if m.Linear == True:
                    nn.init.kaiming_uniform_(m.linear.weight.data, mode='fan_in', nonlinearity='relu')
                    if hasattr(m.linear.bias, 'data'):
                        m.linear.bias.data.fill_(0)
                else:
                    nn.init.kaiming_uniform_(m.conv.weight.data, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(-0.1)
            elif isinstance(m, nn.Conv2d):
                nn.init.dirac_(m.weight.data)
    elif init == 2:
        for m in module.children():
            if isinstance(m, BinBlock):
                if m.Linear == True:
                    nn.init.eye_(m.linear.weight.data)
                    if hasattr(m.linear.bias, 'data'):
                        m.linear.bias.data.fill_(0)
                else:
                    nn.init.dirac_(m.conv.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.eye_(m.weight.data)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(-0.1)
            elif isinstance(m, nn.Conv2d):
                nn.init.dirac_(m.weight.data)
    return


class BinBlock(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
                 Linear=False, previous_conv=False, size=0, bias=False, batch_norm=False):
        super(BinBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.layer_type = 'BinBlock'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.previous_conv = previous_conv
        self.bias = bias
        self.Linear = Linear
        self.size = size
        self.batch_norm = batch_norm
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            if self.batch_norm:
                self.bn = nn.BatchNorm2d(self.input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(self.input_channels, self.output_channels,
                                  kernel_size=self.kernel_size, stride=self.stride, padding=padding, groups=groups,
                                  bias=self.bias)
        else:
            if self.batch_norm:
                if self.previous_conv:
                    self.bn = nn.BatchNorm2d(int(self.input_channels / self.size), eps=1e-4, momentum=0.1, affine=True)
                else:
                    self.bn = nn.BatchNorm1d(self.input_channels, eps=1e-4, momentum=0.1, affine=True)
            else:
                pass
            self.linear = nn.Linear(self.input_channels, output_channels, bias=self.bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.batch_norm:
            x = self.bn(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            if self.previous_conv:
                x = x.flatten(1)  # x.view(x.size(0), self.input_channels)
            x = self.linear(x)
        x = self.relu(x)
        return x
