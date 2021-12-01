import sys

sys.path.append(".")
import network
import numpy as np
import torch
from torch.autograd import grad


def trainingData():
    # ic
    pass


class PINN:
    def __init__(self) -> None:
        self.net = network(2, 2, 4, 20)

    def f(self, xt):
        xt = xt.clone()
        xt.requires_grad = True

        h = self.net(xt)
        u = h[:, 0:1]
        v = h[:, 1:2]
        h_xt = grad(h.sum(), xt, create_graph=True)[0]

        f = self.i * h_t + 0.5 * h_xx + torch.sqrt(u, v) * h
