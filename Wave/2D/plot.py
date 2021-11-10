import sys

sys.path.append("./2D-Wave/")
import torch
from main import PINN
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import meshgrid

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

pinn = PINN()
pinn.net.load_state_dict(torch.load("./2D-Wave/weight.pt"))

# x = np.arange(0, 10, 0.1)
# y = np.arange(0, 10, 0.1)
# t = np.arange(0, 10, 0.1)

# x_, y_, t_ = meshgrid(x, y, t)

# x_ = x_.reshape((-1, 1))
# y_ = y_.reshape((-1, 1))
# t_ = t_.reshape((-1, 1))
# x_ = torch.tensor(x_, dtype=torch.float64).to(device)
# y_ = torch.tensor(y_, dtype=torch.float64).to(device)
# t_ = torch.tensor(t_, dtype=torch.float64).to(device)
# with torch.no_grad():
#     u_pred = pinn.net(x_, y_, t_).cpu().numpy().reshape((100, 100, 100))

x = torch.ones((100, 1), dtype=torch.float64).to(device) * 0.5
y = torch.ones((100, 1), dtype=torch.float64).to(device) * 0.5
t_ = np.linspace(0, 1, 100)[:, None]
t = torch.tensor(t_, dtype=torch.float64).to(device)

u_src = np.sin(t_ * 2 * np.pi * 10)

with torch.no_grad():
    u = pinn.net(x, y, t).cpu().numpy()

# plt.plot(u)
plt.plot(u_src)
plt.show()
