import sys

sys.path.append(".")
from network import DNN
import numpy as np
import torch
from torch.autograd import grad
from pyDOE import lhs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
x_min = 0
x_max = 2
t_min = 0
t_max = 5

ub = np.array([x_max, t_max])
lb = np.array([x_min, t_min])

N_0 = 100
N_bc = 300
N_f = 10000

## IC , u(x,0) = 0
x_0 = np.random.uniform(x_min, x_max, (N_0, 1))
t_0 = np.zeros((N_0, 1))
xt_0 = np.hstack([x_0, t_0])
u_0 = np.where(x_0 < 1, x_0 / 2, 1 - x_0 / 2)
u_t_0 = np.zeros((N_0, 1))

# BC : Fixed end points
x_bc = np.random.choice([x_min, x_max], size=(N_bc, 1))
t_bc = np.random.uniform(t_min, t_max, (N_bc, 1))
xt_bc = np.hstack([x_bc, t_bc])
u_bc = np.zeros((N_bc, 1))

# Collocation points
xt_f = np.random.uniform(lb, ub, (N_f, 2))


# Convert to tensor
xt_0 = torch.tensor(xt_0, dtype=torch.float32).to(device)
u_0 = torch.tensor(u_0, dtype=torch.float32).to(device)
u_t_0 = torch.tensor(u_t_0, dtype=torch.float32).to(device)

xt_bc = torch.tensor(xt_bc, dtype=torch.float32).to(device)
u_bc = torch.tensor(u_bc, dtype=torch.float32).to(device)

xt_f = torch.tensor(xt_f, dtype=torch.float32).to(device)


class PINN:
    c = 1.0

    def __init__(self) -> None:
        self.net = DNN(dim_in=2, dim_out=1, n_layer=5, n_node=40, ub=ub, lb=lb).to(
            device
        )
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.adam = torch.optim.Adam(self.net.parameters())
        self.ms = lambda x: torch.mean(torch.square(x))
        self.iter = 0

    def loss_ic(self, xt):
        xt = xt.clone()
        xt.requires_grad = True
        u = self.net(xt)

        u_t = grad(u.sum(), xt, create_graph=True)[0][:, 1:2]
        mse_0 = self.ms(u - u_0) + self.ms(u_t - u_t_0)
        return mse_0

    def loss_bc(self, xt):
        u = self.net(xt)
        mse_bc = self.ms(u - u_bc)
        return mse_bc

    def loss_pde(self, xt):
        xt = xt.clone()
        xt.requires_grad = True
        u = self.net(xt)

        u_xt = grad(u.sum(), xt, create_graph=True, retain_graph=True)[0]
        u_x = u_xt[:, 0:1]
        u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]

        u_t = u_xt[:, 1:2]
        u_tt = grad(u_t.sum(), xt, create_graph=True)[0][:, 1:2]

        pde = u_tt - (self.c ** 2) * (u_xx)
        mse_pde = self.ms(pde)
        return mse_pde

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()
        mse_0 = self.loss_ic(xt_0)
        mse_bc = self.loss_bc(xt_bc)
        mse_pde = self.loss_pde(xt_f)

        # collocation points
        loss = mse_0 + mse_bc + mse_pde

        loss.backward()
        self.iter += 1
        print(
            f"\r{self.iter}, Loss : {loss.item():.5e}, ic : {mse_0:.3e}, bc : {mse_bc:.3e}, f : {mse_pde:.3e}",
            end="",
        )
        if self.iter % 500 == 0:
            torch.save(self.net.state_dict(), "./Wave/1D/weight.pt")
            print("")

        return loss


if __name__ == "__main__":
    pinn = PINN()
    for i in range(5000):
        pinn.closure()
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "./Wave/1D/weight.pt")
