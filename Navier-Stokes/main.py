"""
PINN Method for the 2D Steady State Incompressible Navier-Stokes Equation
please refer to "problem.md" for problem defenition.
"""


import sys

sys.path.append(".")
from network import DNN
from Utils import *
import numpy as np
import torch
from torch.autograd import grad
from pyDOE import lhs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 0.4
r = 0.05
xc = 0.2
yc = 0.2

ub = np.array([x_max, y_max])
lb = np.array([x_min, y_min])

### Data Prepareation ###
N_b = 200  # inlet & outlet
N_w = 400  # wall
N_s = 200  # surface
N_c = 40000  # collocation
N_r = 10000


def getData():
    # inlet, v=0 & inlet velocity
    inlet_x = np.zeros((N_b, 1))
    inlet_y = np.random.uniform(y_min, y_max, (N_b, 1))
    inlet_u = 4 * inlet_y * (0.4 - inlet_y) / (0.4 ** 2)
    inlet_v = np.zeros((N_b, 1))
    inlet_xy = np.concatenate([inlet_x, inlet_y], axis=1)
    inlet_uv = np.concatenate([inlet_u, inlet_v], axis=1)

    # outlet, p=0
    xy_outlet = np.random.uniform([x_max, y_min], [x_max, y_max], (N_b, 2))

    # wall, u=v=0
    upwall_xy = np.random.uniform([x_min, y_max], [x_max, y_max], (N_w, 2))
    dnwall_xy = np.random.uniform([x_min, y_min], [x_max, y_min], (N_w, 2))
    upwall_uv = np.zeros((N_w, 2))
    dnwall_uv = np.zeros((N_w, 2))

    # cylinder surface, u=v=0
    theta = np.linspace(0.0, 2 * np.pi, N_s)
    cyl_x = (r * np.cos(theta) + xc).reshape(-1, 1)
    cyl_y = (r * np.sin(theta) + yc).reshape(-1, 1)
    cyl_xy = np.concatenate([cyl_x, cyl_y], axis=1)
    cyl_uv = np.zeros((N_s, 2))

    # all boundary except outlet
    xy_bnd = np.concatenate([inlet_xy, upwall_xy, dnwall_xy, cyl_xy], axis=0)
    uv_bnd = np.concatenate([inlet_uv, upwall_uv, dnwall_uv, cyl_uv], axis=0)

    # Collocation
    xy_col = lb + (ub - lb) * lhs(2, N_c)

    # refine points around cylider
    refine_ub = np.array([xc + 2 * r, yc + 2 * r])
    refine_lb = np.array([xc - 2 * r, yc - 2 * r])

    xy_col_refine = refine_lb + (refine_ub - refine_lb) * lhs(2, N_r)
    xy_col = np.concatenate([xy_col, xy_col_refine], axis=0)

    # remove collocation points inside the cylinder

    dst_from_cyl = np.sqrt((xy_col[:, 0] - xc) ** 2 + (xy_col[:, 1] - yc) ** 2)
    xy_col = xy_col[dst_from_cyl > r].reshape(-1, 2)

    # concatenate all xy for collocation
    xy_col = np.concatenate((xy_col, xy_bnd, xy_outlet), axis=0)

    # convert to tensor
    xy_bnd = torch.tensor(xy_bnd, dtype=torch.float32).to(device)
    uv_bnd = torch.tensor(uv_bnd, dtype=torch.float32).to(device)
    xy_outlet = torch.tensor(xy_outlet, dtype=torch.float32).to(device)
    xy_col = torch.tensor(xy_col, dtype=torch.float32).to(device)
    return xy_col, xy_bnd, uv_bnd, xy_outlet


xy_col, xy_bnd, uv_bnd, xy_outlet = getData()


class PINN:
    rho = 1
    mu = 0.02

    def __init__(self) -> None:
        self.net = DNN(dim_in=2, dim_out=6, n_layer=4, n_node=40, ub=ub, lb=lb).to(
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
        self.adam = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        self.losses = {"bc": [], "outlet": [], "pde": []}
        self.iter = 0

    def predict(self, xy):
        out = self.net(xy)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        sig_xx = out[:, 3:4]
        sig_xy = out[:, 4:5]
        sig_yy = out[:, 5:6]
        return u, v, p, sig_xx, sig_xy, sig_yy

    def bc_loss(self, xy):
        u, v = self.predict(xy)[0:2]
        mse_bc = torch.mean(torch.square(u - uv_bnd[:, 0:1])) + torch.mean(
            torch.square(v - uv_bnd[:, 1:2])
        )
        return mse_bc

    def outlet_loss(self, xy):
        out = self.net(xy)
        p = out[:, 1:2]
        mse_outlet = torch.mean(torch.square(p))
        return mse_outlet

    def pde_loss(self, xy):
        xy = xy.clone()
        xy.requires_grad = True
        u, v, p, sig_xx, sig_xy, sig_yy = self.predict(xy)

        u_out = grad(u.sum(), xy, create_graph=True)[0]
        v_out = grad(v.sum(), xy, create_graph=True)[0]
        sig_xx_out = grad(sig_xx.sum(), xy, create_graph=True)[0]
        sig_xy_out = grad(sig_xy.sum(), xy, create_graph=True)[0]
        sig_yy_out = grad(sig_yy.sum(), xy, create_graph=True)[0]

        u_x = u_out[:, 0:1]
        u_y = u_out[:, 1:2]

        v_x = v_out[:, 0:1]
        v_y = v_out[:, 1:2]

        sig_xx_x = sig_xx_out[:, 0:1]
        sig_xy_x = sig_xy_out[:, 0:1]
        sig_xy_y = sig_xy_out[:, 1:2]
        sig_yy_y = sig_yy_out[:, 1:2]

        # continuity equation
        f0 = u_x + v_y

        # navier-stokes equation
        f1 = self.rho * (u * u_x + v * u_y) - sig_xx_x - sig_xy_y
        f2 = self.rho * (u * v_x + v * v_y) - sig_xy_x - sig_yy_y

        # cauchy stress tensor
        f3 = -p + 2 * self.mu * u_x - sig_xx
        f4 = -p + 2 * self.mu * v_y - sig_yy
        f5 = self.mu * (u_y + v_x) - sig_xy

        mse_f0 = torch.mean(torch.square(f0))
        mse_f1 = torch.mean(torch.square(f1))
        mse_f2 = torch.mean(torch.square(f2))
        mse_f3 = torch.mean(torch.square(f3))
        mse_f4 = torch.mean(torch.square(f4))
        mse_f5 = torch.mean(torch.square(f5))
        mse_pde = mse_f0 + mse_f1 + mse_f2 + mse_f3 + mse_f4 + mse_f5

        return mse_pde

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        mse_bc = self.bc_loss(xy_bnd)
        mse_outlet = self.outlet_loss(xy_outlet)
        mse_pde = self.pde_loss(xy_col)
        loss = mse_bc + mse_outlet + mse_pde

        loss.backward()

        self.losses["bc"].append(mse_bc.detach().cpu().item())
        self.losses["outlet"].append(mse_outlet.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        self.iter += 1
        print(
            f"\r{self.iter} Loss: {loss.item():.5e} BC: {mse_bc.item():.3e} outlet: {mse_outlet.item():.3e} pde: {mse_pde.item():.3e}",
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
    torch.save(pinn.net.state_dict(), "./Navier-Stokes/weight.pt")
    plotLoss(pinn.losses, "./Navier-Stokes/loss_curve.png", ["BC", "Outlet", "PDE"])

