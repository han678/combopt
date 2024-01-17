import torch.nn as nn
import numpy
from models import BinBlock


class BinWeights():
    def __init__(self, model):
        count_targets = 0
        self.bin_index = []
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
        return

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True). \
                mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):  # compute \alpha *sign(w)
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            if len(s) == 4:
                m = self.target_modules[index].data.norm(1, 3, keepdim=True) \
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            elif len(s) == 2:
                m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
            self.target_modules[index].data = \
                self.target_modules[index].data.sign().mul(m.expand(s))

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
