import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad


def get_hk(k):
    # Normalizing Factor hk
    k1, k2 = k[0], k[1]
    hk = dblquad(lambda x1, x2: ((np.cos(k1 * np.pi * x1 / 2) ** 2) * np.cos(k2 * np.pi * x2 / 2) ** 2), 0, 2,
                 lambda x1: 0, lambda x1: 2, epsabs=1.49e-01, epsrel=1.49e-01)
    return np.sqrt(hk[0])


def get_Fk(hk, x, k):
    x1, x2, k1, k2 = x[0][0], x[1][0], k[0], k[1]
    Fk = (1 / hk) * (np.cos(k1 * np.pi * x1 / 2)) * (np.cos(k2 * np.pi * x2 / 2))
    return Fk


def get_ck(x_traj, dt, T, k, hk):
    n, _, _ = np.shape(x_traj)
    integral = np.zeros(n)
    for j in range(n):
        Fk_i = get_Fk(hk, x_traj[j], k)
        integral[j] = Fk_i
    ck = (1 / T) * np.trapz(integral, dx=dt)
    return ck


def get_phi_k(Fk):
    phi_k = dblquad(lambda x1, x2:
                    (Fk * (np.linalg.det(2 * np.pi * Sigma)) ** (-0.5)) * (np.exp(-0.5 * (x1 ** 2 + x2 ** 2))),
                    -1, 1, lambda x1: -1, lambda x1: 1, epsabs=1.49e-01, epsrel=1.49e-01)[0]
    return phi_k


size = 7
b_start, b_end,  = 0, size
b_arr, eps_arr = np.linspace(b_start, b_end, size), []
b_ctr = 0

x0 = np.array([[0, 1]]).T

Sigma = np.eye(2)

T, dt = 90, 0.1
samples = T / dt

K = 10

for b in b_arr:
    A = np.array([[0, 1], [-1, -b]])

    x_traj = np.zeros((int(samples + 1), 2, 1))
    x_traj[0] = x0

    # Runge-kutta trajectory creation
    for i in range(1, int(samples + 1)):
        k1 = dt * (A @ x_traj[i - 1])
        k2 = dt * (A @ (x_traj[i - 1] + k1 / 2.))
        k3 = dt * (A @ (x_traj[i - 1] + k2 / 2.))
        k4 = dt * (A @ (x_traj[i - 1] + k3))
        x_traj[i] = x_traj[i - 1] + (1 / 6.) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Print Trajectories
    # plt.scatter(x_traj[:, 0], x_traj[:, 1])
    # plt.show()

    eps = 0
    for k1 in range(1, K):
        for k2 in range(1, K):
            k = [k1, k2]
            hk = get_hk(k)
            Fk = get_Fk(hk, x_traj, k)
            phi_k = get_phi_k(Fk)
            ck = get_ck(x_traj, dt, T, k, hk)
            eps += ((1 + np.linalg.norm(k) ** 2) ** (-3 / 2)) * abs(ck - phi_k) ** 2

    # print(b_ctr)
    # print(eps)
    eps_arr.append(eps)
    b_ctr += 1
