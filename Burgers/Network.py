import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class layer(nn.Module):
    def __init__(self, n_in, n_out, activation=None):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x


class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer(2, 20, nn.Tanh()),
            layer(20, 20, nn.Tanh()),
            layer(20, 20, nn.Tanh()),
            layer(20, 20, nn.Tanh()),
            layer(20, 20, nn.Tanh()),
            layer(20, 20, nn.Tanh()),
            layer(20, 20, nn.Tanh()),
            layer(20, 20, nn.Tanh()),
            layer(20, 20, nn.Tanh()),
            layer(20, 1),
        )

    def forward(self, xt):
        return self.net(xt)
