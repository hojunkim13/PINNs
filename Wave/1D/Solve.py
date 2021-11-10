from Network import DNN
import numpy as np
import torch
from torch.autograd import Variable, grad
from pyDOE import lhs

torch.set_default_dtype(torch.float64)
np.random.seed(1234)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Parameters
lb = {"x": 0, "t": 0}
ub = {"x": 1, "t": 1}

N_u = 500
N_f = 20000


## IC , t=0
x_ic = np.random.uniform(lb["x"], ub["x"], (N_u, 1))
t_ic = np.zeros((N_u, 1))
# u_ic = np.zeros((N_u, 1))
u_ic = np.sin(x_ic * np.pi * 2)

# BC case 1 : Controlled end points, x=0 or x=L
x_bc1 = np.zeros((N_u, 1))
t_bc1 = np.random.uniform(lb["t"], ub["t"], (N_u, 1))
# u_bc1 = np.sin(t_bc1 * 2 * np.pi)
u_bc1 = np.zeros((N_u, 1))

x_bc2 = np.ones((N_u, 1)) * ub["x"]
t_bc2 = np.random.uniform(lb["t"], ub["t"], (N_u, 1))
u_bc2 = np.zeros((N_u, 1))

x_bc = np.vstack((x_bc1, x_bc2))
t_bc = np.vstack((t_bc1, t_bc2))
u_bc = np.vstack((u_bc1, u_bc2))

# collocation points
x_f = lb["x"] + (ub["x"] - lb["x"]) * lhs(1, N_f)
t_f = lb["t"] + (ub["t"] - lb["t"]) * lhs(1, N_f)
x_f = np.vstack((x_ic, x_bc, x_f))
t_f = np.vstack((t_ic, t_bc, t_f))

rand_idx_bc = np.random.choice(len(x_bc), N_u, replace=False)

x_ic = torch.tensor(x_ic, dtype=torch.float64).to(device)
t_ic = torch.tensor(t_ic, dtype=torch.float64).to(device)
u_ic = torch.tensor(u_ic, dtype=torch.float64).to(device)
x_bc = torch.tensor(x_bc[rand_idx_bc], dtype=torch.float64).to(device)
t_bc = torch.tensor(t_bc[rand_idx_bc], dtype=torch.float64).to(device)
u_bc = torch.tensor(u_bc[rand_idx_bc], dtype=torch.float64).to(device)
x_f = torch.tensor(x_f, dtype=torch.float64).to(device)
t_f = torch.tensor(t_f, dtype=torch.float64).to(device)


class PINN:
    c = 1

    def __init__(self) -> None:
        self.net = DNN().to(device)
        self.iter = 0
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=5000,
            max_eval=5000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.adam = torch.optim.Adagrad(self.net.parameters())
        self.mse = lambda x: torch.mean(torch.square(x))

    def f(self, x, t):
        x = Variable(x, requires_grad=True)
        t = Variable(t, requires_grad=True)
        u = self.net(x, t)

        u_x = grad(u.sum(), x, create_graph=True)[0]
        u_xx = grad(u_x.sum(), x, create_graph=True)[0]

        u_t = grad(u.sum(), t, create_graph=True)[0]
        u_tt = grad(u_t.sum(), t, create_graph=True)[0]

        f = u_tt - (self.c ** 2) * (u_xx)
        return f

    def closure(self):
        # IC
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        x = Variable(x_ic, requires_grad=True)
        t = Variable(t_ic, requires_grad=True)

        u_pred_ic = self.net(x, t)
        u_t_pred_ic = grad(u_pred_ic.sum(), t, create_graph=True)[0]

        mse_u_ic = self.mse(u_pred_ic - u_ic) + self.mse(u_t_pred_ic)

        # BC
        u_pred_bc = self.net(x_bc, t_bc)
        mse_u_bc = self.mse(u_pred_bc - u_bc)

        mse_u = mse_u_ic + mse_u_bc

        # collocation points
        f = self.f(x_f, t_f)
        mse_f = self.mse(f)
        loss = mse_u + mse_f

        loss.backward()
        self.iter += 1
        print(
            f"\r{self.iter}, Loss : {loss.item():.5e}, ic : {mse_u_ic:.3e}, bc : {mse_u_bc:.3e}, f : {mse_f:.3e}",
            end="",
        )
        if self.iter % 500 == 0:
            torch.save(self.net.state_dict(), "./Wave/1D/weight.pt")
            print("")

        return loss


if __name__ == "__main__":
    pinn = PINN()
    print("Learning with ADAM optimizer")
    for i in range(5000):
        pinn.closure()
        pinn.adam.step()
    print("Learning with L-BFGS-B optimizer")
    pinn.lbfgs.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "./Wave/1D/weight.pt")
