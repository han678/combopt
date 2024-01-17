import torch
import numpy as np
import warnings


def neg_logistic_loss(y, wx):
    y = y.view(-1, 1)
    wx = wx.view(-1, 1)
    u = -y.mul(wx)
    loss1 = torch.where(u <= 85.0, torch.log(1.0 + torch.exp(u)), torch.tensor(.0))
    loss2 = torch.where(u >= 85.0, u + torch.log(1.0 + torch.exp(-u)), torch.tensor(.0))
    loss = loss1.add(loss2)
    return -torch.sum(loss)


def neg_cross_entropy_loss(y, wx):
    y = y.view(-1, 1)
    wx = wx.view(-1, 1)
    p = torch.sigmoid(wx)
    loss = -y.mul(p) - (1 - y).mul(1 - p)
    return -loss.sum()


#### implement randomized Algorithm
def randomized_submodular(train_x, train_y, p, seed):
    with torch.no_grad():
        n_sample, n_feature = train_x.size()
        # Initialize weights vector: n_feature*1 tensor
        w_A = torch.full((n_feature, 1), fill_value=-p, dtype=torch.float)
        w_B = torch.full((n_feature, 1), fill_value=p, dtype=torch.float)
        wx_A = torch.mm(train_x, w_A)
        wx_B = torch.mm(train_x, w_B)
        k = 1
        for i in range(n_feature):
            # print("Iteration: ", k)
            increasement = train_x[:, i].view(n_sample, -1) * 2.0 * p
            w_A_next = w_A.clone()
            w_A_next[i] = p

            wx_A_next = torch.add(wx_A, increasement)
            a = neg_logistic_loss(train_y, wx_A_next) - neg_logistic_loss(train_y, wx_A)

            w_B_next = w_B.clone()
            w_B_next[i] = -p
            wx_B_next = torch.add(wx_B, -increasement)
            b = neg_logistic_loss(train_y, wx_B_next) - neg_logistic_loss(train_y, wx_B)

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
            np.random.seed(i + seed)  # set seed
            rand = np.random.rand()
            if rand <= marg_prob:
                w_A = w_A_next.clone()
                wx_A = wx_A_next.clone()
            else:
                w_B = w_B_next.clone()
                wx_B = wx_B_next.clone()
            k += 1
    return w_A  # return w_A is a n_feature*1 tensor


def binary_submodular(train_x, train_y, r, p, seed):
    with torch.no_grad():
        n_sample, n_feature = train_x.size()
        # Initialize weights vector: n_feature*1 tensor
        w_A = torch.full((n_feature, 1), fill_value=-p, dtype=torch.float)
        w_B = torch.full((n_feature, 1), fill_value=p, dtype=torch.float)
        wx_A = torch.mm(train_x, w_A) + r
        wx_B = torch.mm(train_x, w_B) + r
        k = 1
        for i in range(n_feature):
            # print("Iteration: ", k)
            increasement = train_x[:, i].view(n_sample, -1) * 2.0 * p
            w_A_next = w_A.clone()
            w_A_next[i] = p

            wx_A_next = torch.add(wx_A, increasement)
            a = neg_logistic_loss(train_y, wx_A_next) - neg_logistic_loss(train_y, wx_A)

            w_B_next = w_B.clone()
            w_B_next[i] = -p
            wx_B_next = torch.add(wx_B, -increasement)
            b = neg_logistic_loss(train_y, wx_B_next) - neg_logistic_loss(train_y, wx_B)

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
            np.random.seed(i + seed)  # set seed
            rand = np.random.rand()
            if rand <= marg_prob:
                w_A = w_A_next.clone()
                wx_A = wx_A_next.clone()
            else:
                w_B = w_B_next.clone()
                wx_B = wx_B_next.clone()
            k += 1
    return w_A  # return w_A is a n_feature*1 tensor


def tenary_submodular(w2, r, train_x, train_y, p, seed):
    # fix w2, update w1
    r2 = -torch.mm(train_x, w2) + r
    w1 = binary_submodular(train_x, train_y, r2, p, seed)
    # then fix w1, update w2
    r1 = torch.mm(train_x, w1) + r
    w2 = binary_submodular(-train_x, train_y, r1, p, seed)
    return w1, w2
