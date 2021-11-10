import torch
from Solve import PINN, ub, lb
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import meshgrid
from celluloid import Camera

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

pinn = PINN()
pinn.net.load_state_dict(torch.load("./Wave/1D/weight.pt"))

x = np.arange(lb["x"], ub["x"], 0.01)
t = np.arange(lb["t"], ub["t"], 0.01)

x_, t_ = meshgrid(x, t)

x_ = x_.reshape((-1, 1))
t_ = t_.reshape((-1, 1))

x_ = torch.tensor(x_, dtype=torch.float64).to(device)
t_ = torch.tensor(t_, dtype=torch.float64).to(device)
with torch.no_grad():
    u_pred = pinn.net(x_, t_).cpu().numpy().reshape((len(t), len(x))).T

print("")
fig, ax = plt.subplots()
cam = Camera(fig)
ax.set_xlabel("$x$")
ax.set_xlabel("$u$")


for i in range(len(t)):
    ax.plot(x, u_pred[:, i], color="black")
    ax.text(0.40, 1.01, f"$t\ =\ {t[i]:.2f}\ [sec]$", transform=ax.transAxes)

    cam.snap()

ani = cam.animate(50, blit=True)
ani.save("./Wave/1D/solution.gif")
