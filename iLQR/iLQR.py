import numpy as np
import matplotlib.pyplot as plt


"""
Initial Trajectory
"""
n = 50
dt = 2 * np.pi / n

u = []
for i in range(n):
    u.extend((1, -0.5))

"""
pos = [[0], [0], [np.pi / 2]]
for i in range(1, n):
    theta = pos[2][i - 1] + u[i][1] * dt
    pos[2].append(theta)
    pos[0].append(pos[0][i - 1] + u[i][0] * dt * np.cos(theta))
    pos[1].append(pos[1][i - 1] + u[i][0] * dt * np.sin(theta))

plt.plot(pos[0], pos[1])
plt.show()
"""


"""
J Function:::
"""
def error(X, X_ref, u, dt):
    X = np.array(X)
    X_ref = np.array(X_ref)
    X[2] *= 0.1
    X_ref[2] *= 0.1
    dist = np.linalg.norm(np.subtract(X, X_ref))
    u_i = np.linalg.norm(u)
    return 0.5 * (dist**2 + u_i**2) * dt


def objective(U):
    x = [0, 0, np.pi / 2]
    U_1 = np.split(U, len(U) / 2)
    n_1 = len(U_1)
    dtt = 2 * np.pi / n_1

    J = 0

    for ii, uu in enumerate(U_1):
        x_ref = [ii * dtt * 4 / (2 * np.pi), 0, ii * dtt * np.pi / 2]
        err = error(x, x_ref, uu, dtt)
        J += err
        x_dot = [uu[0] * np.cos(x[2]), uu[0] * np.sin(x[2]), uu[1]]
        x = np.add(x, x_dot)

    J_3 = J + 0.5 * (np.linalg.norm(U_1[0]))**2 * dtt
    return J_3


"""
Directional Derivative:
"""
# What to do for A and B? !!!!
A = np.array([[0, 1], [-1.6, -0.4]])
B = np.array([[0], [1]])
R = 0.1
P = np.array([[1, 0], [0, 0.01]])

a = [0, 0.5]
b = [0, 1]

BTp = np.dot(-1/R, u)
BTp_split = np.split(BTp, len(BTp) / 2)
for i, vec in enumerate(BTp_split):
    BTp_split[i] = vec + b

v = np.dot(-1/R, BTp_split)

z = np.zeros((np.shape(v)))
for i in range(n-1):
    Bv = np.array([B[0] * v[i][0], B[1] * v[i][1]]).transpose()
    Az = np.matmul(A, z[i])
    z_dot = Az + Bv
    z[i+1] = z[i] + z_dot

# How to calculate pT???
def calc_opt_var(Z, V, pT):



opt_var
"""
e = 0.0001
i = 0
"""
