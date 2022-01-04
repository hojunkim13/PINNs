import sys

sys.path.append(".")
import numpy as np
import torch
from torch.autograd import grad
from network import DNN
from scipy.io import loadmat

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

N_train = 5000

data = loadmat("./Navier-Stokes-identification/cylinder_nektar_wake.mat")
u = data["U_star"][:, 0, :]  # N x T
v = data["U_star"][:, 1, :]  # N x T
p = data["p_star"]  # N x T

x = data["X_star"][:, 0:1]  # N x 1
y = data["X_star"][:, 1:2]  # N x 1
t = data["t"]  # 200 x 1

ub = np.array([x.max(), y.max(), t.max()])
lb = np.array([x.min(), y.min(), t.min()])

x_ = np.tile(x, (1, len(t)))
y_ = np.tile(y, (1, len(t)))
t_ = np.tile(t, (1, N_train)).T

x_ = x_.flatten()[:, None]
y_ = y_.flatten()[:, None]
t_ = t_.flatten()[:, None]
xyt = np.concatenate([x_, y_, t_], axis=1)


u_ = u.flatten()[:, None]
v_ = v.flatten()[:, None]
p_ = p.flatten()[:, None]
uv = np.concatenate([u_, v_], axis=1)

# 1% Noisy Data Preparation
noise = 0.01
noisy_uv = uv + noise * np.std(uv) * np.random.randn(*uv.shape)

idx = np.random.choice(len(xyt), N_train, replace=False)
xyt = torch.tensor(xyt[idx], dtype=torch.float).to(device)
uv = torch.tensor(uv[idx], dtype=torch.float).to(device)
noisy_uv = torch.tensor(noisy_uv[idx], dtype=torch.float32).to(device)

u_star = u_[idx]
v_star = v_[idx]
p_star = p_[idx]


class PINN:
    def __init__(self, uv):
        self.uv = uv
        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)
        self.net = DNN(dim_in=3, dim_out=2, n_layer=8, n_node=20, ub=ub, lb=lb).to(
            device
        )
        self.net.register_parameter("lambda_1", self.lambda_1)
        self.net.register_parameter("lambda_2", self.lambda_2)

        self.adam = torch.optim.Adam(self.net.parameters())
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.iter = 0

    def predict(self, xyt):
        if xyt.requires_grad == False:
            xyt = xyt.clone()
            xyt.requires_grad = True
        psi_p = self.net(xyt)
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]
        u = grad(psi.sum(), xyt, create_graph=True)[0][:, 1:2]
        v = -grad(psi.sum(), xyt, create_graph=True)[0][:, 0:1]
        return u, v, p

    def loss_func(self, xyt):
        xyt = xyt.clone()
        xyt.requires_grad = True
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        u, v, p = self.predict(xyt)

        u_xyt = grad(u.sum(), xyt, create_graph=True)[0]
        u_x = u_xyt[:, 0:1]
        u_y = u_xyt[:, 1:2]
        u_t = u_xyt[:, 2:3]
        u_xx = grad(u_x.sum(), xyt, create_graph=True)[0][:, 0:1]
        u_yy = grad(u_y.sum(), xyt, create_graph=True)[0][:, 1:2]

        v_xyt = grad(v.sum(), xyt, create_graph=True)[0]
        v_x = v_xyt[:, 0:1]
        v_y = v_xyt[:, 1:2]
        v_t = v_xyt[:, 2:3]
        v_xx = grad(v_x.sum(), xyt, create_graph=True)[0][:, 0:1]
        v_yy = grad(v_y.sum(), xyt, create_graph=True)[0][:, 1:2]

        p_xyt = grad(p.sum(), xyt, create_graph=True)[0]
        p_x = p_xyt[:, 0:1]
        p_y = p_xyt[:, 1:2]

        f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
        f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)

        mse_uv = torch.mean(torch.square(u - uv[:, 0:1]) + torch.square(v - uv[:, 1:2]))
        mse_f = torch.mean(torch.square(f_u)) + torch.mean(torch.square(f_v))
        return mse_uv + mse_f

    def closure(self):
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        loss = self.loss_func(xyt)
        loss.backward()

        self.iter += 1
        print(
            f"\r{self.iter} loss : {loss.item():.3e} l1 : {self.lambda_1.item():.5f}, l2 : {self.lambda_2.item():.5f}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")
        return loss


def calcError(pinn):
    u_pred, v_pred, p_pred = pinn.predict(xyt)

    u_pred = u_pred.detach().cpu().numpy()
    v_pred = v_pred.detach().cpu().numpy()
    p_pred = p_pred.detach().cpu().numpy()

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - u_pred, 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - u_pred, 2) / np.linalg.norm(p_star, 2)

    lambda1 = pinn.lambda_1.detach().cpu().item()
    lambda2 = pinn.lambda_2.detach().cpu().item()
    error_lambda1 = np.abs(lambda1 - 1) * 100
    error_lambda2 = np.abs(lambda2 - 0.01) * 100
    print(
        f"\nError u  : {error_u:.3e}%",
        f"\nError v  : {error_v:.3e}%",
        f"\nError p  : {error_p:.3e}%",
        f"\nError l1 : {error_lambda1:.5f}%",
        f"\nError l2 : {error_lambda2:.5f}%",
    )
    return (error_u, error_v, error_p, error_lambda1, error_lambda2)


if __name__ == "__main__":
    nets = []
    for noise, uv_ in enumerate([uv, noisy_uv]):
        pinn = PINN(uv_)
        for i in range(5000):
            loss = pinn.closure()
            pinn.adam.step()
        pinn.lbfgs.step(pinn.closure)
        torch.save(
            pinn.net.state_dict(),
            f"./Navier-Stokes-Identification/weight_noise{noise}.pt",
        )
        nets.append(pinn)

    calcError(nets[0])
    calcError(nets[1])

