from collections import OrderedDict

from torch import nn

from models.BinBlock import BinBlock, initialization


class LeNet5(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, init=1, bias=True):
        super(LeNet5, self).__init__()
        self.bias = bias
        self.init = init
        self.body = nn.Sequential(OrderedDict([
            ('bin_conv1',
             BinBlock(input_channels=input_channels, output_channels=6, kernel_size=5, stride=1, padding=0, bias=self.bias,
                      previous_conv=True,
                      batch_norm=False)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            ('bin_conv2', BinBlock(6, 16, kernel_size=5, stride=1, padding=0, bias=self.bias, previous_conv=True,
                                   batch_norm=False)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            ('bin_fc1', BinBlock(16 * 5 * 5, 120, Linear=True, previous_conv=True, size=5 * 5, bias=self.bias,
                                 batch_norm=False)),
            ('bin_fc2', BinBlock(120, 84, Linear=True, previous_conv=False, bias=self.bias, batch_norm=False)),
        ]))
        self.body.add_module('fc2', nn.Linear(84, num_classes, bias=True))
        initialization(self.init, self.body)

    def forward(self, x):
        for m in self.body.children():
            if isinstance(m, nn.Linear):
                x = x.flatten(1)
            x = m(x)
        return x
