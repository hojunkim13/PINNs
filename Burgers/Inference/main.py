import sys

sys.path.insert(0, ".")

from network import DNN
import numpy as np
import torch
from torch.autograd import grad
from pyDOE import lhs

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


"""
Burgers Eqn.
f = u_t + u*u_x - (0.01 / pi) * u_xx = 0, x ~ [-1, 1], t ~ [0, 1]
u(x, 0) = -sin(pi * x)
u(-1, t) = u(1, t) = 0
"""

N_u = 100
N_f = 10000


x_min = -1
x_max = 1
t_min = 0
t_max = 1

ub = np.array([x_max, t_max])
lb = np.array([x_min, t_min])

x_bc1 = np.random.uniform(x_min, x_max, (N_u, 1))
t_bc1 = np.zeros((N_u, 1))
u_bc1 = -np.sin(np.pi * x_bc1)

x_bc2 = np.ones((N_u, 1))
x_bc2[: N_u // 2] = -1
t_bc2 = np.random.uniform(t_min, t_max, (N_u, 1))
u_bc2 = np.zeros((N_u, 1))

x_bc = np.vstack([x_bc1, x_bc2])
t_bc = np.vstack([t_bc1, t_bc2])
xt_bc = np.hstack([x_bc, t_bc])
u_bc = np.vstack([u_bc1, u_bc2])

rand_idx = np.random.choice(len(xt_bc), N_u, replace=True)

xt_bc = torch.tensor(xt_bc[rand_idx], dtype=torch.float32).to(device)
u_bc = torch.tensor(u_bc[rand_idx], dtype=torch.float32).to(device)

x_f = x_min + (x_max - x_min) * lhs(1, N_f)
t_f = t_min + (t_max - t_min) * lhs(1, N_f)
xt_f = np.hstack([x_f, t_f])

xt_f = torch.tensor(xt_f, dtype=torch.float32).to(device)
xt_f = torch.vstack([xt_f, xt_bc])


class PINN:
    mu = 0.01 / np.pi

    def __init__(self):
        self.net = DNN(dim_in=2, dim_out=1, n_layer=7, n_node=20, ub=ub, lb=lb).to(
            device
        )
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.iter = 0

    def f(self, xt):
        xt = xt.clone()
        xt.requires_grad = True
        u = self.net(xt)

        u_xt = grad(u.sum(), xt, create_graph=True)[0]
        u_x = u_xt[:, 0:1]
        u_t = u_xt[:, 1:2]
        u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]

        f = u_t + u * u_x - self.mu * u_xx
        return f

    def closure(self):
        self.optimizer.zero_grad()

        u_pred = self.net(xt_bc)
        mse_u = torch.mean(torch.square(u_pred - u_bc))

        f = self.f(xt_f)
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
    torch.save(pinn.net.state_dict(), "./Burgers/Inference/weight.pt")
