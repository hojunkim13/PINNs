import sys

sys.path.insert(0, "./Burgers")
import torch
from Burgers import PINN, ub, lb, x_bc, t_bc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat

pgf_with_latex = {  # setup matplotlib to use latex for output
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}
plt.rcParams.update(pgf_with_latex)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

x_bc = x_bc.cpu().numpy()
t_bc = t_bc.cpu().numpy()


pinn = PINN()
pinn.net.load_state_dict(torch.load("./Burgers/Burgers_inference/weight.pt"))

x = np.arange(lb["x"], ub["x"], 0.01)
t = np.arange(lb["t"], ub["t"], 0.01)

x, t = np.meshgrid(x, t)
x_ = np.reshape(x, (-1, 1))
t_ = np.reshape(t, (-1, 1))

x_ = torch.tensor(x_, dtype=torch.float64).to(device)
t_ = torch.tensor(t_, dtype=torch.float64).to(device)
with torch.no_grad():
    u_pred = pinn.net(torch.hstack((x_, t_)))

u_pred = u_pred.cpu().numpy().reshape(t.shape).T

############## Plot 1 ###############
fig = plt.figure(figsize=(9, 5))
gs0 = GridSpec(1, 2, figure=fig)
gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)

ax = fig.add_subplot(gs0[0, :])
im = ax.imshow(
    u_pred, cmap="rainbow", extent=[lb["t"], ub["t"], lb["x"], ub["x"]], aspect="auto"
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

plt.colorbar(im, cax=cax, label="$u(x,t)$")
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
ax.set_title(r"$u(x,t)$", fontsize=10)
bc_pt = ax.scatter(
    t_bc,
    x_bc,
    s=15,
    c="black",
    marker="x",
    label=f"Data ({len(t_bc):d} points)",
    clip_on=False,
)
ax.legend(frameon=False, loc="best")

############## Plot 2 ###############
x = np.arange(lb["x"], ub["x"], 0.01)
t = np.arange(lb["t"], ub["t"], 0.01)
data = loadmat("./Burgers/burgers_shock.mat")
x_ = data["x"]
u_sol = data["usol"]


gs1 = GridSpec(1, 3, figure=fig)
gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

t_slice = [0.25, 0.5, 0.75]
axes = []
for i in range(3):
    ax = fig.add_subplot(gs1[0, i])
    ax.plot(x_, u_sol[:, int(t_slice[i] * 100)], "b-", linewidth=2, label="Exact")
    ax.plot(x, u_pred[:, int(t_slice[i] * 100)], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x,t)$")
    ax.set_title(f"$t = {t_slice[i]}s$", fontsize=10)
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    axes.append(ax)
axes[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=10
)

fig.savefig(
    "./Burgers/Burgers_inference/Burgers_sol.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=500,
)
