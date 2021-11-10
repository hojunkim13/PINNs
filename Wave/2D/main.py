import sys

sys.path.append("./2D-Wave/")
from Network import DNN
import numpy as np
import torch
from torch.autograd import Variable, grad
from pyDOE import lhs

torch.set_default_dtype(torch.float64)
np.random.seed(42)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Parameters
ub = 1.0
lb = 0.0

N_u = 500
N_f = 20000


########
x_points = np.random.uniform(lb, ub, (N_u, 1))
y_points = np.random.uniform(lb, ub, (N_u, 1))
t_points = np.random.uniform(lb, ub, (N_u, 1))

# source
freq = 10  # 10 times
x_src = np.ones((N_u, 1)) * ub / 2
y_src = np.ones((N_u, 1)) * ub / 2
t_src = t_points.copy()
u_src = np.sin(t_src * 2 * np.pi * freq)

x_src = torch.tensor(x_src, dtype=torch.float64).to(device)
y_src = torch.tensor(y_src, dtype=torch.float64).to(device)
t_src = torch.tensor(t_src, dtype=torch.float64).to(device)
u_src = torch.tensor(u_src, dtype=torch.float64).to(device)

## BC
x_bc1 = np.zeros((N_u, 1))
y_bc1 = y_points.copy()
t_bc1 = t_points.copy()
u_bc1 = np.zeros((N_u, 1))

x_bc2 = np.ones((N_u, 1)) * ub
y_bc2 = y_points.copy()
t_bc2 = t_points.copy()
u_bc2 = np.zeros((N_u, 1))

x_bc3 = x_points.copy()
y_bc3 = np.zeros((N_u, 1))
t_bc3 = t_points.copy()
u_bc3 = np.zeros((N_u, 1))

x_bc4 = x_points.copy()
y_bc4 = np.ones((N_u, 1)) * ub
t_bc4 = t_points.copy()
u_bc4 = np.zeros((N_u, 1))

x_bc_o0 = np.vstack([x_bc1, x_bc2, x_bc3, x_bc4])
y_bc_o0 = np.vstack([y_bc1, y_bc2, y_bc3, y_bc4])
t_bc_o0 = np.vstack([t_bc1, t_bc2, t_bc3, t_bc4])
u_bc_o0 = np.vstack([u_bc1, u_bc2, u_bc3, u_bc4])

x_case0 = np.zeros((N_u, 1))
y_case0 = y_points.copy()
t_case0 = t_points.copy()

x_case1 = np.ones((N_u, 1)) * ub
y_case1 = y_points.copy()
t_case1 = t_points.copy()


x_case2 = x_points.copy()
y_case2 = np.zeros((N_u, 1))
t_case2 = t_points.copy()

x_case3 = x_points.copy()
y_case3 = np.ones((N_u, 1)) * ub
t_case3 = t_points.copy()

x_case0 = torch.tensor(x_case0, dtype=torch.float64).to(device)
y_case0 = torch.tensor(y_case0, dtype=torch.float64).to(device)
t_case0 = torch.tensor(t_case0, dtype=torch.float64).to(device)
x_case1 = torch.tensor(x_case1, dtype=torch.float64).to(device)
y_case1 = torch.tensor(y_case1, dtype=torch.float64).to(device)
t_case1 = torch.tensor(t_case1, dtype=torch.float64).to(device)
x_case2 = torch.tensor(x_case2, dtype=torch.float64).to(device)
y_case2 = torch.tensor(y_case2, dtype=torch.float64).to(device)
t_case2 = torch.tensor(t_case2, dtype=torch.float64).to(device)
x_case3 = torch.tensor(x_case3, dtype=torch.float64).to(device)
y_case3 = torch.tensor(y_case3, dtype=torch.float64).to(device)
t_case3 = torch.tensor(t_case3, dtype=torch.float64).to(device)

x_case = [x_case0, x_case1, x_case2, x_case3]
y_case = [x_case0, y_case1, y_case2, y_case3]
t_case = [x_case0, t_case1, t_case2, t_case3]

rand_idx = np.random.choice(len(x_bc_o0), N_u, replace=False)

x_bc_o0_ts = torch.tensor(x_bc_o0[rand_idx], dtype=torch.float64).to(device)
y_bc_o0_ts = torch.tensor(y_bc_o0[rand_idx], dtype=torch.float64).to(device)
t_bc_o0_ts = torch.tensor(t_bc_o0[rand_idx], dtype=torch.float64).to(device)
u_bc_o0_ts = torch.tensor(u_bc_o0[rand_idx], dtype=torch.float64).to(device)


# collocation points
col_pts = lb + (ub - lb) * lhs(3, N_f)
x_f = col_pts[:, 0:1]
y_f = col_pts[:, 1:2]
t_f = col_pts[:, 2:3]


x_f = torch.tensor(x_f, dtype=torch.float64).to(device)
y_f = torch.tensor(y_f, dtype=torch.float64).to(device)
t_f = torch.tensor(t_f, dtype=torch.float64).to(device)


class PINN:
    c = 1

    def __init__(self) -> None:
        self.net = DNN().to(device)
        self.iter = 0
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=5000,
            max_eval=5000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

    def f(self, x, y, t):
        x = Variable(x, requires_grad=True)
        y = Variable(y, requires_grad=True)
        t = Variable(t, requires_grad=True)
        u = self.net(x, y, t)

        u_x = grad(u.sum(), x, create_graph=True)[0]
        u_xx = grad(u_x.sum(), x, create_graph=True)[0]

        u_y = grad(u.sum(), y, create_graph=True)[0]
        u_yy = grad(u_y.sum(), y, create_graph=True)[0]

        u_t = grad(u.sum(), t, create_graph=True)[0]
        u_tt = grad(u_t.sum(), t, create_graph=True)[0]

        f = u_tt - self.c ** 2 * (u_xx + u_yy)
        return f

    def closure(self):
        u_src_pred = self.net(x_src, y_src, t_src)
        mse_u_src = torch.mean(torch.square(u_src_pred - u_src))

        u_o0_pred = self.net(x_bc_o0_ts, y_bc_o0_ts, t_bc_o0_ts)
        mse_u_o0 = torch.mean(torch.square(u_o0_pred - u_bc_o0_ts))

        mse_u_o1s = []
        for i in range(len(x_case)):
            x = Variable(x_case[i], requires_grad=True)
            y = Variable(y_case[i], requires_grad=True)
            t = Variable(t_case[i], requires_grad=True)

            u = self.net(x, y, t)

            u_x = grad(u.sum(), x, create_graph=True)[0]
            u_y = grad(u.sum(), y, create_graph=True)[0]
            u_t = grad(u.sum(), t, create_graph=True)[0]

            if i == 0:
                v = u_x - self.c * u_t
            elif i == 1:
                v = u_x + self.c * u_t
            elif i == 2:
                v = u_y - self.c * u_t
            else:
                v = u_y + self.c * u_t
            mse_value = torch.mean(torch.square(v))
            mse_u_o1s.append(mse_value)

        mse_u_o1 = sum(mse_u_o1s) / 4
        mse_u = (mse_u_src + mse_u_o0 + mse_u_o1) / 3
        f = self.f(x_f, y_f, t_f)
        mse_f = torch.mean(torch.square(f))
        loss = mse_u + mse_f

        self.optimizer.zero_grad()
        loss.backward()
        self.iter += 1
        print(f"\r{self.iter}, Loss : {loss.item():.5e}", end="")
        if self.iter % 500 == 0:
            print("")

        return loss


if __name__ == "__main__":
    pinn = PINN()
    pinn.optimizer.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "./2D-Wave/weight.pt")
