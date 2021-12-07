import torch
from main import PINN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pinn = PINN()
pinn.net.load_state_dict(torch.load("./Wave/2D/weight.pt"))

x_min = 0.0
x_max = 1.0
y_min = 0.0
y_max = 1.0
t_min = 0.0
t_max = 1.0

x = np.arange(x_min, x_max, 0.01)
y = np.arange(y_min, y_max, 0.01)
t = np.arange(t_min, t_max, 0.01)

X, Y, T = np.meshgrid(x, y, t)
x_ = X.reshape(-1, 1)
y_ = Y.reshape(-1, 1)
t_ = T.reshape(-1, 1)

xyt = np.hstack([x_, y_, t_])
xyt = torch.tensor(xyt, dtype=torch.float).to(device)

u = pinn.net(xyt).detach().cpu().numpy().reshape(X.shape)
# u = np.sin(2 * np.pi * X) * np.sin(np.pi * Y)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(u.max() * 1.1, u.min() * 1.1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u(x,t)$")
ax.set_title("$2D\ Wave$")

plot = [
    ax.plot_surface(
        X[..., 0], Y[..., 0], u[..., 0], vmin=u.min(), vmax=u.max(), cmap="magma",
    )
]
cb = fig.colorbar(plot[0], ax=ax, fraction=0.046, pad=0.2)


def update(frame, plot):
    ax.view_init(elev=30, azim=frame)
    plot[0].remove()
    plot[0] = ax.plot_surface(
        X[..., 0], Y[..., 0], u[..., frame], vmin=u.min(), vmax=u.max(), cmap="magma",
    )


ani = FuncAnimation(fig, update, frames=len(t), fargs=(plot,))
ani.save("./Wave/2D/solution.gif", fps=15)
