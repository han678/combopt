from collections import OrderedDict

from torch import nn

from models.BinBlock import BinBlock, initialization


class CNN6(nn.Module):
    def __init__(self, bias=False, input_channels=3, num_classes=1, init=1):
        super(CNN6, self).__init__()
        self.bias = bias
        self.init = init
        self.body = nn.Sequential(OrderedDict([
            ('bin_conv1',
             BinBlock(input_channels=input_channels, output_channels=12, kernel_size=4, stride=2, padding=2,
                      bias=self.bias, previous_conv=True, Linear=False, batch_norm=False)),
            ('bin_conv2', BinBlock(12, 36, kernel_size=3, stride=2, padding=1, bias=self.bias, previous_conv=True,
                                   Linear=False, batch_norm=False)),
            ('bin_conv3', BinBlock(36, 36, kernel_size=3, stride=1, padding=1, bias=self.bias, previous_conv=True,
                                   Linear=False, batch_norm=False)),
            ('bin_conv4', BinBlock(36, 36, kernel_size=3, stride=1, padding=1, bias=self.bias, previous_conv=True,
                                   Linear=False, batch_norm=False)),
            ('bin_conv5', BinBlock(36, 64, kernel_size=3, stride=2, padding=1, bias=self.bias, previous_conv=True,
                                   Linear=False, batch_norm=False)),
            ('bin_conv6', BinBlock(64, 128, kernel_size=3, stride=1, padding=1, bias=self.bias, previous_conv=True,
                                   Linear=False, batch_norm=False)),
        ]))
        self.body.add_module('fc1', nn.Linear(3200, num_classes, bias=True))
        initialization(self.init, self.body)

    def forward(self, x):
        for m in self.body:
            if isinstance(m, nn.Linear):
                x = x.flatten(1)
            x = m(x)
        return x
