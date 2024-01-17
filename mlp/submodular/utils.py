import math
import warnings
import numpy as np
import torch
from models import BinBlock
import torch.nn as nn


def y_times_w(y, w):
    y = y.unsqueeze(dim=1)
    if len(w.shape) == 2 or len(w.shape) == 1:
        yw = torch.mul(y, w)
    elif len(w.shape) == 4:
        yw = torch.mul(torch.repeat_interleave(y, repeats=w.shape[1], dim=1).unsqueeze(dim=2).unsqueeze(dim=3), w)
    return yw


def compute_pi(Wx, yw):
    m = nn.ReLU()
    if len(Wx.shape) == 2 or len(Wx.shape) == 1:
        pi = torch.sum(torch.mul(yw, m(Wx)), dim=1)
    elif len(Wx.shape) == 4:
        pi = torch.sum(torch.mul(yw, m(Wx)), dim=[1, 2, 3])
    return pi.unsqueeze(dim=1)


def tightest_convex_loss(ywi, zi, ri):
    device = ri.device
    ywi, ri = ywi.to(device), ri.to(device)
    m = nn.ReLU()
    loss = torch.full((ywi.shape[0], 1), fill_value=0.0).to(device)
    # condition 1: y * wi <= 0
    condition1 = (ywi <= 0).to(device)
    loss = torch.where(condition1.reshape(-1, 1), exp_part_loss(-torch.mul(ywi, m(zi)) - ri, device), loss)

    # condition 2: y * wi > 0 and zi >= 0
    condition2 = torch.logical_and((zi >= 0).reshape(1, -1), (ywi > 0).reshape(1, -1)).to(device)
    loss = torch.where(condition2.reshape(-1, 1), exp_part_loss(-torch.mul(ywi, m(zi)) - ri, device), loss)

    # condition 3: y * wi > 0 and zi < 0
    condition3 = torch.logical_and((zi < 0).reshape(1, -1), (ywi > 0).reshape(1, -1)).to(device)
    loss = torch.where(condition3.reshape(-1, 1), torch.add(-0.5 * torch.mul(ywi, zi), exp_part_loss(-ri, device)),
                       loss)
    return loss


def exp_part_loss(u, device):
    c1 = torch.where(u <= 85.0, torch.log(1.0 + torch.exp(u)), torch.tensor(.0).to(device))
    c2 = torch.where(u >= 85.0, u + torch.log(1.0 + torch.exp(-u)), torch.tensor(.0).to(device))
    c = c1.add(c2)
    return c


def neg_loss(ywi, Z, ri):
    if len(ywi.shape) == 2:  # fully connected layer
        loss = tightest_convex_loss(ywi=ywi, zi=Z, ri=ri)
        return -torch.sum(loss).item()
    elif len(ywi.shape) == 4:
        device = Z.device
        m = nn.ReLU()
        ywi_pos = m(ywi)  # max(0,ywi)
        ywi_neg = torch.where(ywi < 0, ywi, torch.tensor(0.).to(device))  # min(0,ywi)
        p1 = compute_v(ywi_pos, Z, ri)
        p2 = compute_v(ywi_neg, m(Z), ri)
        loss1 = exp_part_loss(-p1, device)
        loss2 = exp_part_loss(-p2, device)
        loss = loss1.add(loss2).div(2)
        return -loss.sum().item()


def compute_v(ywi, phi, ri):
    v = 2 * torch.sum(torch.mul(ywi, phi), dim=[1, 2, 3]).unsqueeze(dim=1) + ri
    return v


def ComputeZ(W, module, x):
    device = x.device
    if module.Linear == True:
        W_init = module.linear.weight.data
    else:
        W_init = module.conv.weight.data
    module = module.to(device)
    W = W.view(W_init.shape)
    W_init.copy_(W)
    with torch.no_grad():
        for m in module.children():
            if not isinstance(m, torch.nn.ReLU):
                x = m(x)
    return x


# implement randomized Algorithm on the single row of W1
def submodularFC(x, ywi, ri, beta):
    with torch.no_grad():
        n_sample, n_feature = x.size()
        device = x.device
        # Initialize weights vector: n_feature*1 tensor
        Wi_A = torch.full((n_feature, 1), fill_value=-beta, dtype=torch.float).to(device)
        Wi_B = torch.full((n_feature, 1), fill_value=beta, dtype=torch.float).to(device)
        Wix_A = torch.mm(x, Wi_A)
        Wix_B = torch.mm(x, Wi_B)
        k = 1
        for i in range(n_feature):
            # print("Iteration: ", k)
            increasement = x[:, i].view(n_sample, -1) * beta
            Wi_A_next = Wi_A.clone()
            Wi_A_next[i] = beta
            Wix_A_next = torch.add(Wix_A, 2. * increasement)
            neg_loss_A = neg_loss(ywi, Wix_A, ri)
            neg_loss_next_A = neg_loss(ywi, Wix_A_next, ri)
            a = neg_loss_next_A - neg_loss_A
            Wi_B_next = Wi_B.clone()
            Wi_B_next[i] = -beta
            Wix_B_next = torch.add(Wix_B, -2. * increasement)
            neg_loss_B = neg_loss(ywi, Wix_B, ri)
            neg_loss_next_B = neg_loss(ywi, Wix_B_next, ri)
            b = neg_loss_next_B - neg_loss_B
            # print("a: ", a, " b: ", b)
            new_a = max(a, 0)
            new_b = max(b, 0)
            # compute marginal probability
            if (new_a + new_b) == 0.0:
                warnings.warn('Both a and b equal to zero.')
                marg_prob = 1 / 2
            else:
                marg_prob = new_a / (new_a + new_b)
            # sample based on marginal probability
            np.random.seed(i + 30)  # set seed
            rand = np.random.rand()
            if rand <= marg_prob:
                Wi_A = Wi_A_next.clone()
                Wix_A = Wix_A_next.clone()
            else:
                Wi_B = Wi_B_next.clone()
                Wix_B = Wix_B_next.clone()
            k += 1
    return Wi_A


def submodularConv(x, m, beta, ywi, Wi, ri):
    device = x.device
    Wi_zero = torch.zeros_like(Wi.unsqueeze(dim=0)).to(device)
    Wi_A = Wi_zero.view(Wi_zero.shape[0], -1).clone()
    Wi_B = Wi_A.clone()
    Wi_A[0][:] = -beta
    Wi_B[0][:] = beta
    oneChannelConv = BinBlock(input_channels=m.input_channels, output_channels=1, kernel_size=m.kernel_size,
                              stride=m.stride, padding=m.padding, bias=m.bias, previous_conv=m.previous_conv).to(device)
    Z_A = ComputeZ(W=Wi_A, module=oneChannelConv, x=x)
    Z_B = ComputeZ(W=Wi_B, module=oneChannelConv, x=x)
    for j in range(Wi_A.shape[1]):
        W_movement = Wi_zero.view(Wi_zero.shape[0], -1).clone()
        W_movement[0][j] = beta
        Wi_A_next = torch.add(Wi_A, 2. * W_movement)
        Wi_B_next = torch.add(Wi_B, -2. * W_movement)
        increasement = ComputeZ(W=W_movement, module=oneChannelConv, x=x)
        Z_A_next = torch.add(Z_A, 2. * increasement)
        Z_B_next = torch.add(Z_B, -2. * increasement)
        neg_loss_A = neg_loss(ywi, Z_A, ri)
        neg_loss_A_next = neg_loss(ywi, Z_A_next, ri)
        neg_loss_B = neg_loss(ywi, Z_B, ri)
        neg_loss_B_next = neg_loss(ywi, Z_B_next, ri)
        a = neg_loss_A_next - neg_loss_A
        b = neg_loss_B_next - neg_loss_B
        new_a = max(a, 0)
        new_b = max(b, 0)
        if (new_a + new_b) == 0.0:
            warnings.warn('Both a and b equal to zero.')
            marg_prob = 1 / 2
        else:
            marg_prob = new_a / (new_a + new_b)

        np.random.seed(j + 30)  # set seed
        rand = np.random.rand()
        if rand <= marg_prob:
            Z_A = Z_A_next.clone()
            Wi_A = Wi_A_next.clone()
        else:
            Z_B = Z_B_next.clone()
            Wi_B = Wi_B_next.clone()
        return Wi_A