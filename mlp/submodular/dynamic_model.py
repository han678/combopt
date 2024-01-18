from collections import OrderedDict
import torch
import torch.nn as nn


class dynamic_model():
    def __init__(self, layer_idx, model):
        super(dynamic_model).__init__()
        self.pretrained = model
        net1_list = list(self.pretrained.body.named_children())[0:layer_idx] if layer_idx != 0 else list()
        net2_list = list(self.pretrained.body.named_children())[layer_idx]
        net3_list = list(self.pretrained.body.named_children())[(layer_idx + 1):]
        self.net1 = nn.Sequential(OrderedDict(net1_list))
        self.net2 = nn.Sequential(OrderedDict([net2_list]))
        self.net3 = nn.Sequential(OrderedDict(net3_list))
        self.initialize_weight(self.net1)
        self.initialize_weight(self.net2)
        self.initialize_weight(self.net3)

    def initialize_weight(self, net):
        pretrain_dict = self.pretrained.body.state_dict()
        net_dict = net.state_dict()
        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in net_dict}
        # overwrite entries in the existing state dict
        net_dict.update(pretrained_dict)
        # load the new state dict
        net.load_state_dict(net_dict)

    def forward(self, x):
        with torch.no_grad():
            if list(self.net1) == [] and hasattr(list(self.net2)[0], 'linear'):
                x = x.flatten(1)
            for layer in self.net1.children():
                if isinstance(layer, nn.Linear):
                    x = x.flatten(1)
                x = layer(x)
            return x

    def forward_res(self, x):
        for layer in self.net2.children():
            if isinstance(layer, nn.Linear):
                x = x.flatten(1)
            phi = layer(x)
        pred = self.forward_net3(phi)
        return phi, pred

    def forward_net3(self, x):
        for m in self.net3:
            if isinstance(m, nn.Linear):
                x = x.flatten(1)
            x = m(x)
        return x