import torch
from main import PINN, xt_ic, xt_lb, xt_ub
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""
pgf_with_latex = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


def test():
    data = loadmat("./Schrodinger/NLS.mat")
    Exact_x = data["x"].flatten()[:, None]
    Exact_t = data["tt"].flatten()[:, None]
    x_mesh, t_mesh = np.meshgrid(Exact_x, Exact_t)
    Exact_xt = np.hstack([x_mesh.flatten()[:, None], t_mesh.flatten()[:, None]])
    Exact_xt = torch.tensor(Exact_xt, dtype=torch.float).to(device)

    Exact = data["uu"]
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]

    with torch.no_grad():
        u_pred, v_pred = pinn.net_uv(Exact_xt)
        h_pred = torch.sqrt(u_pred ** 2 + v_pred ** 2)

    u_pred = u_pred.cpu().numpy()
    v_pred = v_pred.cpu().numpy()
    h_pred = h_pred.cpu().numpy()
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
    print(f"Error_u : {error_u:.3e}")
    print(f"Error_v : {error_v:.3e}")
    print(f"Error_h : {error_h:.3e}")


plt.rcParams.update(pgf_with_latex)

pinn = PINN()
pinn.net.load_state_dict(torch.load("./Schrodinger/weight.pt"))

xt_ic = xt_ic.cpu().numpy()
xt_lb = xt_lb.cpu().numpy()
xt_ub = xt_ub.cpu().numpy()

train_data_xt = np.vstack([xt_ic, xt_lb, xt_ub])

data = loadmat("./Schrodinger/NLS.mat")
x = data["x"].flatten()
t = data["tt"].flatten()
X, T = np.meshgrid(x, t)
X = X.flatten()[:, None]
T = T.flatten()[:, None]
u_sol = data["uu"].T
h_sol = np.abs(u_sol)


with torch.no_grad():
    xt = torch.tensor(np.hstack([X, T]), dtype=torch.float).to(device)
    uv_pred = pinn.net(xt)
    h_pred = torch.sqrt(uv_pred[:, 0] ** 2 + uv_pred[:, 1] ** 2)

h_pred = h_pred.cpu().numpy().reshape(u_sol.shape).T

test()
############## Plot 1 ###############

fig = plt.figure(figsize=(10, 6))
gs0 = GridSpec(1, 2, figure=fig)
gs0.update(top=1 - 0.06, bottom=0.66, left=0.15, right=0.85, wspace=0)

ax = fig.add_subplot(gs0[0, :])
im = ax.imshow(h_pred, cmap="YlGnBu", extent=[0, np.pi / 2, -5, 5], aspect="auto")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.2)

plt.colorbar(im, cax=cax, label="$|h(x,t)|$")
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")

ax.set_title("$|h(x,t)|$")

bc_pt = ax.scatter(
    train_data_xt[:, 1],
    train_data_xt[:, 0],
    s=15,
    c="black",
    marker="x",
    label=f"Data ({len(train_data_xt):d} points)",
    clip_on=False,
)
ax.legend(frameon=False, loc="best")

############## Plot 2 ###############


gs1 = GridSpec(1, 3, figure=fig)
gs1.update(top=0.9, bottom=0, left=0.15, right=0.85, wspace=0.5)

t_idx = [75, 100, 125]
axes = []
for i in range(3):
    ax = fig.add_subplot(gs1[0, i])
    ax.plot(x, h_sol[t_idx[i], :], "b-", linewidth=2, label="Exact")
    ax.plot(x, h_pred[:, t_idx[i]], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x,t)$")
    ax.set_title(f"$t = {t[t_idx[i]]:.2f}s$", fontsize=10)
    ax.axis("square")
    ax.set_xlim([-5, 5])
    ax.set_ylim([0, 5])
    axes.append(ax)
axes[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.50), ncol=2, frameon=False, fontsize=10
)

fig.savefig(
    "./Schrodinger/solution.png", dpi=500, bbox_inches="tight", pad_inches=0,
)
