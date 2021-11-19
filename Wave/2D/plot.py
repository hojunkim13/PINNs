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

x_mesh, y_mesh = np.meshgrid(x, y)
x_ = x_mesh.reshape(-1, 1)
y_ = y_mesh.reshape(-1, 1)

u_seq = []

for dt in t:
    t_ = np.ones_like(x_) * dt
    xyt = np.hstack([x_, y_, t_])
    xyt = torch.tensor(xyt, dtype=torch.float).to(device)
    u = pinn.net(xyt).detach().cpu().numpy().reshape(len(y), len(x))
    u_seq.append(u)

u_min = np.min(u_seq)
u_max = np.max(u_seq)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(u_min, u_max)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$u(x,t)$")
ax.set_title("$2D\ Wave$")
u = u_seq[0]
plot = [ax.plot_surface(x_mesh, y_mesh, u, vmin=u_min, vmax=u_max, cmap="magma")]
cb = fig.colorbar(plot[0], ax=ax, fraction=0.046, pad=0.2)


def update(frame, plot):
    azim = 30 + int(frame / len(t) * 180)
    ax.view_init(elev=30, azim=azim)
    u = u_seq[frame]
    plot[0].remove()
    plot[0] = ax.plot_surface(x_mesh, y_mesh, u, vmin=u_min, vmax=u_max, cmap="magma",)


ani = FuncAnimation(fig, update, frames=u.shape[-1], fargs=(plot,))
ani.save("./Wave/2D/solution.gif", fps=20)
