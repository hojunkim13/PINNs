import sys

sys.path.append(".")
from network import DNN
import numpy as np
import torch
from torch.autograd import Variable, grad
from pyDOE import lhs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
x_min = 0
x_max = 1
t_min = 0
t_max = 1

N_u = 200
N_f = 10000


## IC , u(x,0) = 0
x_ic = np.random.uniform(x_min, x_max, (N_u, 1))
t_ic = np.zeros((N_u, 1))
# u_ic = np.zeros((N_u, 1))
u_ic = np.sin(x_ic * np.pi)
xt_ic = np.hstack([x_ic, t_ic])

# BC : Controlled end points, u(x,0) = u(x,L) = 0
# BC : Controlled end points, u(x,0) = sin(2pi * t * freq), u(x, L) = 0
x_bc1 = np.zeros((N_u, 1))
t_bc1 = np.random.uniform(t_min, t_max, (N_u, 1))
u_bc1 = np.zeros((N_u, 1))
# u_bc1 = np.sin(2 * np.pi * t_bc1 * 2)

x_bc2 = np.ones((N_u, 1)) * x_max
t_bc2 = np.random.uniform(t_min, t_max, (N_u, 1))
u_bc2 = np.zeros((N_u, 1))

x_bc = np.vstack([x_bc1, x_bc2])
t_bc = np.vstack([t_bc1, t_bc2])
xt_bc = np.hstack([x_bc, t_bc])
u_bc = np.vstack((u_bc1, u_bc2))

# collocation points
x_f = x_min + (x_max - x_min) * lhs(1, N_f)
t_f = t_min + (t_max - t_min) * lhs(1, N_f)
x_f = np.vstack([x_ic, x_bc, x_f])
t_f = np.vstack([t_ic, t_bc, t_f])


xt_ic = torch.tensor(xt_ic, dtype=torch.float32).to(device)
u_ic = torch.tensor(u_ic, dtype=torch.float32).to(device)

xt_bc = torch.tensor(xt_bc, dtype=torch.float32).to(device)
u_bc = torch.tensor(u_bc, dtype=torch.float32).to(device)

x_f = torch.tensor(x_f, dtype=torch.float32).to(device)
t_f = torch.tensor(t_f, dtype=torch.float32).to(device)
xt_f = torch.hstack([x_f, t_f])


class PINN:
    c = 1

    def __init__(self) -> None:
        self.net = DNN(
            dim_in=2, dim_out=1, n_layer=5, n_node=20, activation=torch.nn.Tanh()
        ).to(device)
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=10000,
            max_eval=10000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.ms = lambda x: torch.mean(torch.square(x))
        self.iter = 0

    def f(self, xt):
        x = Variable(xt[:, 0:1], requires_grad=True)
        t = Variable(xt[:, 1:2], requires_grad=True)
        u = self.net(torch.hstack([x, t]))

        u_x = grad(u.sum(), x, create_graph=True)[0]
        u_xx = grad(u_x.sum(), x, create_graph=True)[0]

        u_t = grad(u.sum(), t, create_graph=True)[0]
        u_tt = grad(u_t.sum(), t, create_graph=True)[0]

        f = u_tt - (self.c ** 2) * (u_xx)
        return f

    def closure(self):
        # IC
        self.optimizer.zero_grad()

        x = Variable(xt_ic[:, 0:1], requires_grad=True)
        t = Variable(xt_ic[:, 1:2], requires_grad=True)

        u_pred_ic = self.net(torch.hstack([x, t]))
        u_t_pred_ic = grad(u_pred_ic.sum(), t, create_graph=True)[0]

        mse_u_ic = self.ms(u_pred_ic - u_ic) + self.ms(u_t_pred_ic)

        # BC
        u_pred_bc = self.net(xt_bc)
        mse_u_bc = self.ms(u_pred_bc - u_bc)

        mse_u = mse_u_ic + mse_u_bc

        # collocation points
        f = self.f(xt_f)
        mse_f = self.ms(f)
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
    pinn.optimizer.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "./Wave/1D/weight.pt")
