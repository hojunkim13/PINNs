import sys

sys.path.insert(0, "./Burgers")

import numpy as np
import torch
from torch.autograd import grad, Variable
from pyDOE import lhs
from Network import DNN

torch.set_default_dtype(torch.float64)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Burgers Eqn.
f = u_t + u*u_x - (0.01 / pi) * u_xx = 0, x ~ [-1, 1], t ~ [0, 1]
u(x, 0) = -sin(pi * x)
u(-1, t) = u(1, t) = 0
"""
N_u = 100
N_f = 10000

lb = {"x": -1, "t": 0}
ub = {"x": 1, "t": 1}
x_bc1 = lb["x"] + (ub["x"] - lb["x"]) * lhs(1, N_u)
t_bc1 = np.zeros((N_u, 1))
u_bc1 = -np.sin(np.pi * x_bc1)

x_bc2 = np.ones((N_u, 1))
t_bc2 = lb["t"] + (ub["t"] - lb["t"]) * lhs(1, N_u)
u_bc2 = np.zeros((N_u, 1))

x_bc3 = -np.ones((N_u, 1))
t_bc3 = lb["t"] + (ub["t"] - lb["t"]) * lhs(1, N_u)
u_bc3 = np.zeros((N_u, 1))

x_bc = np.vstack([x_bc1, x_bc2, x_bc3])
t_bc = np.vstack([t_bc1, t_bc2, t_bc3])
u_bc = np.vstack([u_bc1, u_bc2, u_bc3])

rand_idx = np.random.choice(len(x_bc), N_u, replace=True)

x_bc = torch.tensor(x_bc[rand_idx], dtype=torch.float64).to(device)
t_bc = torch.tensor(t_bc[rand_idx], dtype=torch.float64).to(device)
u_bc = torch.tensor(u_bc[rand_idx], dtype=torch.float64).to(device)

x_f = lb["x"] + (ub["x"] - lb["x"]) * lhs(1, N_f)
t_f = lb["t"] + (ub["t"] - lb["t"]) * lhs(1, N_f)
x_f = torch.tensor(x_f, dtype=torch.float64).to(device)
t_f = torch.tensor(t_f, dtype=torch.float64).to(device)
x_f = torch.vstack((x_f, x_bc))
t_f = torch.vstack((t_f, t_bc))


class PINN:
    def __init__(self):
        self.net = DNN()
        self.net.to(device)
        # self.optimizer = torch.optim.Adam(self.net.parameters()) # Adam optimizer
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # can be "strong_wolfe"
        )
        self.iter = 0

    def f(self, x, t):
        mu = 0.01 / np.pi

        x = Variable(x, requires_grad=True).to(device)
        t = Variable(t, requires_grad=True).to(device)
        u = self.net(torch.cat((x, t), dim=1))

        u_t = grad(u.sum(), t, create_graph=True)[0]
        u_x = grad(u.sum(), x, create_graph=True)[0]
        u_xx = grad(u_x.sum(), x, create_graph=True)[0]

        f = u_t + u * u_x - mu * u_xx
        return f

    def closure(self):
        self.optimizer.zero_grad()

        u_pred = self.net(torch.cat((x_bc, t_bc), dim=1))

        mse_u = torch.mean(torch.square(u_pred - u_bc))

        f = self.f(x_f, t_f)
        mse_f = torch.mean(torch.square(f))

        loss = mse_u + mse_f
        loss.backward()

        self.iter += 1
        print(f"\r{self.iter} loss : {loss.item():.3e}", end="")
        if self.iter % 500 == 0:
            print("")

        return loss


if __name__ == "__main__":
    pinn = PINN()
    pinn.optimizer.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "./Burgers/Burgers_inference/weight.pt")
