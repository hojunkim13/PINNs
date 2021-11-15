import torch
from solve import PINN, x_min, x_max, t_min, t_max
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pinn = PINN()
pinn.net.load_state_dict(torch.load("./Wave/1D/weight.pt"))

x = np.arange(x_min, x_max, 0.01)
t = np.arange(t_min, t_max, 0.01)

x_mesh, t_mesh = np.meshgrid(x, t)

x_ = x_mesh.reshape((-1, 1))
t_ = t_mesh.reshape((-1, 1))

x_ts = torch.tensor(x_, dtype=torch.float32).to(device)
t_ts = torch.tensor(t_, dtype=torch.float32).to(device)
xt_ts = torch.hstack([x_ts, t_ts])
with torch.no_grad():
    u_pred = pinn.net(xt_ts).cpu().numpy().reshape(t_mesh.shape).T


fig, ax = plt.subplots()
ax.set_xlabel("$x$")
ax.set_xlabel("$u$")
ax.set_xlim([x_min - 0.1, x_max + 0.1])
ax.set_ylim([u_pred.min() - 0.1, u_pred.max() + 0.1])
(line,) = ax.plot([], [], color="k")
text = ax.text(0.40, 1.01, "", transform=ax.transAxes)


def update(frame):
    line.set_data(x, u_pred[:, frame])
    text.set_text(f"$t\ =\ {t[frame]:.2f}\ [sec]$")


ani = FuncAnimation(fig, update, frames=len(t), interval=50)
ani.save("./Wave/1D/solution.gif", dpi=300)
