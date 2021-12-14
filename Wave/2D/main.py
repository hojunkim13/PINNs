import sys

sys.path.append(".")
from network import DNN
from Utils import *
import numpy as np
import torch
from torch.autograd import grad
from pyDOE import lhs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0
t_min = 0.0
t_max = 1.0

ub = np.array([x_max, y_max, t_max])
lb = np.array([x_min, y_min, t_min])

N_0 = 200
N_bc = 200
N_f = 20000


########


x_points = lambda n: np.random.uniform(x_min, x_max, (n, 1))
y_points = lambda n: np.random.uniform(y_min, y_max, (n, 1))
t_points = lambda n: np.random.uniform(t_min, t_max, (n, 1))

# IC
x_0 = x_points(N_0)
y_0 = y_points(N_0)
t_0 = np.zeros((N_0, 1))
xyt_0 = np.hstack([x_0, y_0, t_0])

u_0 = np.sin(2 * np.pi * x_0) * np.sin(np.pi * y_0)
u_t_0 = np.zeros((N_0, 1))

## BC
x_bc1 = np.zeros((N_bc, 1))
y_bc1 = y_points(N_bc)
t_bc1 = t_points(N_bc)
u_bc1 = np.zeros((N_bc, 1))

x_bc2 = np.ones((N_bc, 1)) * x_max
y_bc2 = y_points(N_bc)
t_bc2 = t_points(N_bc)
u_bc2 = np.zeros((N_bc, 1))

x_bc3 = x_points(N_bc)
y_bc3 = np.zeros((N_bc, 1))
t_bc3 = t_points(N_bc)
u_bc3 = np.zeros((N_bc, 1))

x_bc4 = x_points(N_bc)
y_bc4 = np.ones((N_bc, 1)) * y_max
t_bc4 = t_points(N_bc)
u_bc4 = np.zeros((N_bc, 1))

x_bc = np.vstack([x_bc1, x_bc2, x_bc3, x_bc4])
y_bc = np.vstack([y_bc1, y_bc2, y_bc3, y_bc4])
t_bc = np.vstack([t_bc1, t_bc2, t_bc3, t_bc4])
xyt_bc = np.hstack([x_bc, y_bc, t_bc])
u_bc = np.vstack([u_bc1, u_bc2, u_bc3, u_bc4])


# collocation points
x_f = x_min + (x_max - x_min) * lhs(1, N_f)
y_f = y_min + (y_max - y_min) * lhs(1, N_f)
t_f = t_min + (t_max - t_min) * lhs(1, N_f)
xyt_f = np.hstack([x_f, y_f, t_f])
xyt_f = np.vstack([xyt_f, xyt_bc, xyt_0])


xyt_0 = torch.tensor(xyt_0, dtype=torch.float32).to(device)
u_0 = torch.tensor(u_0, dtype=torch.float32).to(device)
u_t_0 = torch.tensor(u_t_0, dtype=torch.float32).to(device)
xyt_bc = torch.tensor(xyt_bc, dtype=torch.float32).to(device)
u_bc = torch.tensor(u_bc, dtype=torch.float32).to(device)
xyt_f = torch.tensor(xyt_f, dtype=torch.float32).to(device)


class PINN:
    c = 1

    def __init__(self) -> None:
        self.net = DNN(dim_in=3, dim_out=1, n_layer=6, n_node=40, ub=ub, lb=lb).to(
            device
        )
        self.iter = 0
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
        self.adam = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        self.losses = {"ic": [], "bc": [], "pde": []}

    def loss_ic(self, xyt):
        xyt = xyt.clone()
        xyt.requires_grad = True

        u = self.net(xyt)
        u_t = grad(u.sum(), xyt, create_graph=True)[0][:, 2:3]

        mse_0 = torch.mean(torch.square(u - u_0)) + torch.mean(
            torch.square(u_t - u_t_0)
        )
        return mse_0

    def loss_bc(self, xyt):
        u = self.net(xyt)
        mse_bc = torch.mean(torch.square(u - u_bc))
        return mse_bc

    def loss_pde(self, xyt):
        xyt = xyt.clone()
        xyt.requires_grad = True

        u = self.net(xyt)

        u_xyt = grad(u.sum(), xyt, create_graph=True)[0]
        u_xx = grad(u_xyt[:, 0:1].sum(), xyt, create_graph=True)[0][:, 0:1]
        u_yy = grad(u_xyt[:, 1:2].sum(), xyt, create_graph=True)[0][:, 1:2]
        u_tt = grad(u_xyt[:, 2:3].sum(), xyt, create_graph=True)[0][:, 2:3]
        pde = u_tt - (self.c ** 2) * (u_xx + u_yy)

        mse_pde = torch.mean(torch.square(pde))
        return mse_pde

    def closure(self):
        self.optimizer.zero_grad()
        self.adam.zero_grad()

        mse_0 = self.loss_ic(xyt_0)
        mse_bc = self.loss_bc(xyt_bc)
        mse_pde = self.loss_pde(xyt_f)

        loss = mse_0 + mse_bc + mse_pde

        loss.backward()

        self.losses["ic"].append(mse_0.detach().cpu().item())
        self.losses["bc"].append(mse_bc.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        self.iter += 1
        print(
            f"\r{self.iter} Loss: {loss.item():.5e} IC: {mse_0.item():.3e} BC: {mse_bc.item():.3e} pde: {mse_pde.item():.3e}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")
        return loss


if __name__ == "__main__":
    pinn = PINN()
    for i in range(10000):
        pinn.closure()
        pinn.adam.step()
    pinn.optimizer.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "./Wave/2D/weight.pt")
    plotLoss(pinn.losses, "./Wave/2d/loss_curve.png")
