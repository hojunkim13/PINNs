import sys

sys.path.append(".")
from network import DNN
import numpy as np
import torch
from torch.autograd import grad
from pyDOE import lhs
import matplotlib.pyplot as plt

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

N_ic = 500
N_bc = 400
N_f = 20000


def normalize(xyt):
    xyt = xyt.clone()
    lb = torch.tensor([x_min, y_min, t_min], dtype=torch.float).to(device)
    ub = torch.tensor([x_max, y_max, t_max], dtype=torch.float).to(device)
    xyt = (xyt - lb) / (ub - lb)
    return xyt


########


x_points = lambda n: np.random.uniform(x_min, x_max, (n, 1))
y_points = lambda n: np.random.uniform(y_min, y_max, (n, 1))
t_points = lambda n: np.random.uniform(t_min, t_max, (n, 1))

# IC
x_ic = x_points(N_ic)
y_ic = y_points(N_ic)
t_ic = np.zeros((N_ic, 1))
xyt_ic = np.hstack([x_ic, y_ic, t_ic])

u_ic = np.sin(2 * np.pi * x_ic) * np.sin(np.pi * y_ic)

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
xyt_f = np.vstack([xyt_f, xyt_bc, xyt_ic])


xyt_ic = torch.tensor(xyt_ic, dtype=torch.float32).to(device)
u_ic = torch.tensor(u_ic, dtype=torch.float32).to(device)
xyt_bc = torch.tensor(xyt_bc, dtype=torch.float32).to(device)
u_bc = torch.tensor(u_bc, dtype=torch.float32).to(device)
xyt_f = torch.tensor(xyt_f, dtype=torch.float32).to(device)


class PINN:
    c = 1

    def __init__(self) -> None:
        self.net = DNN(
            dim_in=3, dim_out=1, n_layer=6, n_node=40, ub=ub, lb=lb
        ).to(device)
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
        self.losses = {"ic": [], "bc": [], "pde": []}
        self.adam = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def u(self, xyt):
        return self.net(normalize(xyt))

    def f(self, xyt):
        g = xyt.clone()
        g.requires_grad = True

        u = self.u(g)

        u_xyt = grad(u.sum(), g, create_graph=True)[0]
        u_xx = grad(u_xyt[:, 0:1].sum(), g, create_graph=True)[0][:, 0:1]
        u_yy = grad(u_xyt[:, 1:2].sum(), g, create_graph=True)[0][:, 1:2]
        u_tt = grad(u_xyt[:, 2:3].sum(), g, create_graph=True)[0][:, 2:3]
        f = u_tt - (self.c ** 2) * (u_xx + u_yy)
        return f

    def closure(self):
        self.optimizer.zero_grad()
        self.adam.zero_grad()

        u_ic_pred = self.u(xyt_ic)
        mse_ic = torch.mean(torch.square(u_ic_pred - u_ic))

        u_bc_pred = self.u(xyt_bc)
        mse_bc = torch.mean(torch.square(u_bc_pred - u_bc))

        f_pred = self.f(xyt_f)
        mse_f = torch.mean(torch.square(f_pred))

        loss = mse_ic + mse_bc + mse_f

        loss.backward()

        self.losses["ic"].append(mse_ic.detach().cpu().item())
        self.losses["bc"].append(mse_bc.detach().cpu().item())
        self.losses["pde"].append(mse_f.detach().cpu().item())
        self.iter += 1
        print(
            f"\r{self.iter} Loss: {loss.item():.5e} IC: {mse_ic.item():.3e} BC: {mse_bc.item():.3e} pde: {mse_f.item():.3e}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")

        return loss


def plotLoss(losses_dict):
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 6))
    axes[0].set_yscale("log")
    for i, j in zip(range(3), ["IC", "BC", "PDE"]):
        axes[i].plot(losses_dict[j.lower()])
        axes[i].set_title(j)
    plt.show()
    fig.savefig("./Wave/2d/loss_curve.png")


if __name__ == "__main__":
    pinn = PINN()
    for i in range(5000):
        pinn.closure()
        pinn.adam.step()
    pinn.optimizer.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "./Wave/2D/weight.pt")
    plotLoss(pinn.losses)
