import sys

sys.path.append(".")

from network import DNN
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from torch.autograd import Variable
from matplotlib.animation import FuncAnimation

dtype = torch.float32  # Or torch.float64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_default_dtype(dtype)
torch.manual_seed(1234)
np.random.seed(1234)


class PINN:
    c = 1

    def __init__(self) -> None:
        self.net = DNN(
            dim_in=2, dim_out=1, n_layer=4, n_node=20, activation=nn.Tanh()
        ).to(device)
        self.optimizer = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            history_size=50,
            max_eval=10000,
            max_iter=10000,
        )
        self.iter = 0

    def f(self, xt):
        x = Variable(xt[:, 0:1], requires_grad=True)
        t = Variable(xt[:, 1:2], requires_grad=True)

        u = self.net(torch.cat([x, t], dim=1))
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        f = u_t - self.c ** 2 * u_xx
        return f

    def closure(self):
        self.optimizer.zero_grad()
        u_pred = self.net(xt_u)
        mse_u = torch.mean(torch.square(u_pred - u_u))

        f_pred = self.f(xt_f)
        mse_f = torch.mean(torch.square(f_pred))
        loss = mse_u + mse_f
        loss.backward()
        self.iter += 1
        print(f"\r{self.iter} loss : {loss.item():.4e}", end="")
        if self.iter % 500 == 0:
            print("")
        return loss


N_u = 100
N_f = 10000

x_min = 0.0
x_max = 1.0
t_min = 0.0
t_max = 1.0

# initial condition, u(x, 0) = f(x)
x_ic = np.random.uniform(low=x_min, high=x_max, size=(N_u, 1))
t_ic = np.zeros((N_u, 1))
u_ic = np.ones((N_u, 1))

# Case 1. Const End, u(0, t) = T
# Case 2. Insulated End, u_x(0, t) = 0
# Case 3. Radiating End, u_x(0, t) = A * u(x0, t)

x_bc = np.zeros((N_u, 1))
t_bc = np.random.uniform(low=t_min, high=t_max, size=(N_u, 1))
u_bc = np.zeros((N_u, 1))

x_u = np.concatenate((x_ic, x_bc), axis=0)
t_u = np.concatenate((t_ic, t_bc), axis=0)
xt_u = np.concatenate((x_u, t_u), axis=1)
u_u = np.concatenate((u_ic, u_bc), axis=0)

x_f = np.random.uniform(x_min, x_max, (N_f, 1))
t_f = np.random.uniform(t_min, t_max, (N_f, 1))
xt_f = np.hstack([x_f, t_f])
xt_f = np.vstack([xt_f, xt_u])

xt_u = torch.tensor(xt_u, dtype=dtype).to(device)
u_u = torch.tensor(u_u, dtype=dtype).to(device)
xt_f = torch.tensor(xt_f, dtype=dtype).to(device)


pinn = PINN()
pinn.optimizer.step(pinn.closure)

####################### Plot ########################

x = np.arange(x_min, x_max, 0.01)
t = np.arange(t_min, t_max, 0.01)
ms_x, ms_t = np.meshgrid(x, t)

x = ms_x.reshape(-1, 1)
t = ms_t.reshape(-1, 1)

x = torch.tensor(x, dtype=dtype).to(device)
t = torch.tensor(t, dtype=dtype).to(device)
xt = torch.hstack([x, t])
u = pinn.net(xt)
u = u.detach().cpu().numpy()
u = u.reshape(ms_t.shape).T


fig, axes = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
bar_x = np.linspace(x_min, x_max, u.shape[0])
bar_y = np.zeros_like(bar_x)
axes[0].set_title("$1D\ Heat\ Equation$")
axes[0].set_yticks([])

im0 = axes[0].scatter(
    bar_x,
    bar_y,
    c=u[:, 0],
    marker="s",
    cmap="rainbow",
    lw=0,
    vmin=u.min(),
    vmax=u.max(),
)
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, extend="neither")
textbox = offsetbox.AnchoredText("", loc=1)
axes[0].add_artist(textbox)

(im1,) = axes[1].plot([], [], color="k")
axes[1].set_ylim(u.min() - u.max(), u.max())
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("$Temperature\ u(x,t)$")


def update(frame):
    temp = u[:, frame]
    im0.set_array(temp)
    im1.set_data(bar_x, temp)
    textbox = offsetbox.AnchoredText(f"{ms_t[frame][0]:.2f} sec", loc=1)
    axes[0].add_artist(textbox)


ani = FuncAnimation(fig, update, frames=range(len(ms_t)), interval=50)
ani.save("./Heat/solution.gif", dpi=300)
