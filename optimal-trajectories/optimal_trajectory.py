import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def error(X, X_ref, u, dt):
    """
    Calculates error based on position
    :param X: Current position
    :param X_ref: Reference Position
    :param u: Current velocity
    :param dt: Time change
    :return: Error from X to X_ref
    """
    X = np.array(X)
    X_ref = np.array(X_ref)
    X[2] *= 0.1
    X_ref[2] *= 0.1
    dist = np.linalg.norm(np.subtract(X, X_ref))
    u_i = np.linalg.norm(u)
    return 0.5 * (dist ** 2 + u_i ** 2) * dt


def objective(U):
    """
    Generates objective function based on control signal
    :param U: Array of control signal
    :return: Objective function J
    """
    x = [0, 0, np.pi / 2]
    U_1 = np.split(U, len(U) / 2)
    n = len(U_1)
    dt = 2 * np.pi / n

    J = 0

    for i, u in enumerate(U_1):
        x_ref = [i * dt * 4 / (2 * np.pi), 0, i * dt * np.pi / 2]
        err = error(x, x_ref, u, dt)
        J += err
        x_dot = [u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]]
        x = np.add(x, x_dot)

    J_3 = J + 0.5 * (np.linalg.norm(U_1[0])) ** 2 * dt
    return J_3


if __name__ == "__main__":
    iterations = 100
    U0 = []
    for it in range(iterations):
        U0.extend((0, 0))

    sol = minimize(objective, np.array(U0), options={'maxiter': 1000})

    # Extracting solution arrays from the minimizer
    u_arr = sol['x']
    u_arr_1 = np.split(u_arr, len(u_arr) / 2)

    x_arr = [[0], [0], [np.pi / 2]]
    for j, u_j in enumerate(u_arr_1):
        x_arr_dot = [u_j[0] * np.cos(x_arr[2][-1]), u_j[0] * np.sin(x_arr[2][-1]), u_j[1]]
        x_arr[0].append(x_arr[0][-1] + x_arr_dot[0])
        x_arr[1].append(x_arr[1][-1] + x_arr_dot[1])
        x_arr[2].append(x_arr[2][-1] + x_arr_dot[2])

    # Plot Finalized Trajectory
    plt.plot(x_arr[0], x_arr[1])
    plt.show()

    # Plot Control Signal
    u0 = []
    u1 = []
    for i, u_val in enumerate(u_arr):
        if not (i % 2):
            u0.append(u_val)
        else:
            u1.append(u_val)

    t_plot = np.linspace(0, 2 * np.pi, np.shape(u0)[0])

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t_plot, u0)
    axs[1].plot(t_plot, u1)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('u0')
    axs[1].set_ylabel('u1')
    fig.tight_layout()
    plt.show()
