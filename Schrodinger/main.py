import sys

sys.path.append(".")
from network import DNN
from Utils import *
import numpy as np
import torch
from torch.autograd import grad
from pyDOE import lhs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x_min = -5.0
x_max = 5.0
t_min = 0
t_max = np.pi / 2

ub = np.array([x_max, t_max])
lb = np.array([x_min, t_min])

N_ic = 50
N_bc = 50
N_f = 20000


def trainingData():
    # ic
    x_ic = np.random.uniform(x_min, x_max, (N_ic, 1))
    t_ic = np.zeros((N_ic, 1))
    xt_ic = np.hstack([x_ic, t_ic])
    u_ic = 2 * 2 / (np.exp(x_ic) + np.exp(-x_ic))  # 2 * sech(x)
    v_ic = np.zeros((N_ic, 1))
    uv_ic = np.hstack([u_ic, v_ic])

    # bc
    t_b = np.random.uniform(t_min, t_max, (N_bc, 1))
    x_lb = np.ones((N_bc, 1)) * x_min
    xt_lb = np.hstack([x_lb, t_b])

    x_ub = np.ones((N_bc, 1)) * x_max
    xt_ub = np.hstack([x_ub, t_b])

    # pde
    xt_f = lb + (ub - lb) * lhs(2, N_f)
    xt_f = np.vstack([xt_ic, xt_lb, xt_ub, xt_f])

    # tensor
    xt_ic = torch.tensor(xt_ic, dtype=torch.float).to(device)
    uv_ic = torch.tensor(uv_ic, dtype=torch.float).to(device)
    xt_lb = torch.tensor(xt_lb, dtype=torch.float).to(device)
    xt_ub = torch.tensor(xt_ub, dtype=torch.float).to(device)
    xt_f = torch.tensor(xt_f, dtype=torch.float).to(device)

    return xt_ic, uv_ic, xt_lb, xt_ub, xt_f


xt_ic, uv_ic, xt_lb, xt_ub, xt_f = trainingData()


class PINN:
    def __init__(self) -> None:
        self.net = DNN(dim_in=2, dim_out=2, n_layer=5, n_node=100, ub=ub, lb=lb).to(
            device
        )
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        self.adam = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()
        self.losses = {"ic": [], "bc": [], "pde": []}
        self.iter = 0

    def net_uv(self, xt):
        uv = self.net(xt)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        return u, v

    def ic_loss(self, xt):
        uv_ic_pred = self.net(xt)
        mse_ic = self.loss_fn(uv_ic_pred, uv_ic)
        return mse_ic

    def bc_loss(self, xt_lb, xt_ub):
        xt_lb = xt_lb.clone()
        xt_ub = xt_ub.clone()
        xt_lb.requires_grad = True
        xt_ub.requires_grad = True

        u_lb, v_lb = self.net_uv(xt_lb)
        u_ub, v_ub = self.net_uv(xt_ub)

        mse_bc1_u = self.loss_fn(u_lb, u_ub)
        mse_bc1_v = self.loss_fn(v_lb, v_ub)

        u_x_lb = grad(u_lb.sum(), xt_lb, create_graph=True)[0][:, 0:1]
        u_x_ub = grad(u_ub.sum(), xt_ub, create_graph=True)[0][:, 0:1]

        v_x_lb = grad(v_lb.sum(), xt_lb, create_graph=True)[0][:, 0:1]
        v_x_ub = grad(v_ub.sum(), xt_ub, create_graph=True)[0][:, 0:1]
        mse_bc2_u = self.loss_fn(u_x_lb, u_x_ub)
        mse_bc2_v = self.loss_fn(v_x_lb, v_x_ub)

        mse_bc = mse_bc1_u + mse_bc1_v + mse_bc2_u + mse_bc2_v
        return mse_bc

    def pde_loss(self, xt):
        xt = xt.clone()
        xt.requires_grad = True
        u, v = self.net_uv(xt)

        u_xt = grad(u.sum(), xt, create_graph=True)[0]
        u_x = u_xt[:, 0:1]
        u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        u_t = u_xt[:, 1:2]

        v_xt = grad(v.sum(), xt, create_graph=True)[0]
        v_x = v_xt[:, 0:1]
        v_xx = grad(v_x.sum(), xt, create_graph=True)[0][:, 0:1]
        v_t = v_xt[:, 1:2]

        f_u = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u  # u
        f_v = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v  # v
        f_target = torch.zeros_like(f_u)

        mse_pde = self.loss_fn(f_u, f_target) + self.loss_fn(f_v, f_target)
        return mse_pde

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()
        mse_ic = self.ic_loss(xt_ic)
        mse_bc = self.bc_loss(xt_lb, xt_ub)
        mse_pde = self.pde_loss(xt_f)
        loss = mse_ic + mse_bc + mse_pde
        loss.backward()

        self.losses["ic"].append(mse_ic.detach().cpu().item())
        self.losses["bc"].append(mse_bc.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        self.iter += 1
        print(
            f"\r{self.iter} Loss: {loss.item():.5e} IC: {mse_ic.item():.3e} BC: {mse_bc.item():.3e} pde: {mse_pde.item():.3e}",
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
    pinn.lbfgs.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "./Schrodinger/weight.pt")
    plotLoss(pinn.losses, "./Schrodinger/loss_curve.png")

