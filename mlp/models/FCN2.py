from collections import OrderedDict

from torch import nn

from models.BinBlock import BinBlock, initialization


class FCN2(nn.Module):
    def __init__(self, bias=False, input_channels=3, n_hidden=100, num_classes=1, init=1):
        super(FCN2, self).__init__()
        self.bias = bias
        self.init = init
        self.body = nn.Sequential(OrderedDict([
            ('bin_fc1',
             BinBlock(input_channels * 32 * 32, n_hidden, Linear=True, previous_conv=True, size=32 * 32, bias=self.bias,
                      batch_norm=False)),
        ]))
        self.body.add_module('fc2', nn.Linear(n_hidden, num_classes, bias=True))
        initialization(self.init, self.body)

    def forward(self, x):
        for m in self.body:
            if isinstance(m, nn.Linear):
                x = x.flatten(1)
            x = m(x)
        return x
