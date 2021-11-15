import torch
import torch.nn as nn
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(1234)
np.random.seed(1234)


class layer(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x


class DNN(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer, n_node, activation=nn.Tanh()):
        super().__init__()

        self.net = nn.ModuleList()
        self.net.append(layer(dim_in, n_node, activation))
        for _ in range(n_layer):
            self.net.append(layer(n_node, n_node, activation))
        self.net.append(layer(n_node, dim_out, activation=None))
        self.net.apply(weights_init)

    def forward(self, xt):
        out = xt
        for layer in self.net:
            out = layer(out)
        return out


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)
