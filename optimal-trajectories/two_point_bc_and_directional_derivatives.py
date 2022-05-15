import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def objective(t, xp):
    # Objective function for 2 point boundary condition problem
    x = xp[:2]
    p = xp[2:]
    A = np.array([[0, 1], [-1.6, -0.4]])
    B = np.array([[0], [1]])
    R = np.array([0.1])
    Q = np.array([[2, 0], [0, 0.01]])

    x_dot = np.matmul(A, x) - np.matmul(np.matmul(B / R, B.T), p)
    p_dot = -1 * (np.matmul(Q, x) + np.matmul(A.T, p))

    return np.vstack((x_dot, p_dot))


def bc(xp_0, xp_1):
    # Setting points for boundary conditions
    return [xp_0[0] - 10, xp_0[1], xp_1[2], xp_1[3]]


# Setting parameters for the 2 point boundary value problems
n = 201
t_in = np.linspace(0, 10, n)
x_in = np.zeros((4, t_in.size))

P1 = np.array([[1, 0], [0, 0.01]])
x_T = np.array([[0], [0]])
p_in = np.matmul(P1.T, x_T)
p_in = p_in.flatten()

x_in[0][0] = 10
x_in[2][-1] = p_in[0]
x_in[3][-1] = p_in[1]

xp_sol = integrate.solve_bvp(objective, bc, t_in, x_in)


# Extracting and plotting the solutions
xp_out = xp_sol['y']
x_out = xp_out[:2]
p_out = xp_out[2:]

u_conv = -1 * 10 * np.array([[0], [1]]).T
u = np.matmul(u_conv, p_out)
u = u.T

t_plot = xp_sol['x']

fig, axs = plt.subplots(3, 1)
axs[0].plot(t_plot, u)
axs[1].plot(t_plot, x_out[0])
axs[2].plot(t_plot, x_out[1])
axs[0].set_xlabel('time')
axs[0].set_ylabel('u')
axs[1].set_ylabel('x_0')
axs[2].set_ylabel('x_1')
fig.tight_layout()
plt.show()


# Computing Directional Derivatives
for iteration in range(10):
    # Directions chosen will be along a sin wave
    t_d = np.linspace(0, 10, n)
    a = b = c = d = np.random.uniform()
    v = a * np.sin(b * t_d + c) + d

    z = np.array([[0], [0]])

    A_dd = np.array([[0, 1], [-1.6, -0.4]])
    B_dd = np.array([[0], [1]])
    Q_dd = np.array([[2, 0], [0, 0.01]])
    R_dd = np.array([0.1])

    integrate_sum = 0

    z_dot = 0

    # Looping through to calculate v and z, and directional derivatives via integrals
    for i in range(n):
        Add_z = np.matmul(A_dd, z)
        B_v = B_dd * v[i]
        z_dot = Add_z + B_v
        x_TQz = np.matmul(np.matmul(x_out.T[i], Q_dd), z)
        uTRV = u[i] * R_dd * v[i]
        integrate_sum += np.trapz(x_TQz, dx=t_d[1]) + np.trapz(x_TQz, dx=t_d[1])
        z = z + (z_dot * t_d[1])

    integrate_sum += np.matmul(np.matmul(x_out.T[-1], P1), z)


