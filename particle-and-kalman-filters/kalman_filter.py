import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


def predict(x, P, F, Q):
    """
    Predicts a priori state estimate and estimate covariance
    """
    x_pred = F @ x
    p_pred = F @ P @ F.T + Q
    return x_pred, p_pred


def update_step(x, P, H, z, R):
    """
    Updates the position and covariance matrix to the current step
    """
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ S
    x_update = x + K @ y
    p_update = (np.eye(2, 2) - K @ H) @ P
    # y_new = z - np.dot(H, x)
    return x_update, p_update


# Kalman Filter
# x is represented by position, there is no control input
variance = 0.1
dt = 0.1

x = np.array([[1., 1.]]).T
F = np.array([[0., 1.], [-1., 0.]])

Fk = expm(F*dt)

Q = np.array([[variance, 0.0], [0.0, variance]])

H = np.array([[1., 0.]])
R = np.array([variance])

P = np.eye(2, 2) * 1.0

# Creates an array for outputs
# Actual reflects the real trajectory
x_actual = np.array([1., 1.])
x_actual_out = np.array([1, 1])

# Obs reflects the trajectory observed with noise
x_obs = np.array([1, 1])
x_obs_out = np.array([1, 1])

for i in range(10):
    # Generates the actual and observed trajectory
    x_actual = Fk @ x_actual
    x_actual_out = np.vstack((x_actual_out, x_actual))

    wk = np.random.normal(0, np.sqrt(0.1))
    x_obs = x_obs + dt * Fk @ x_obs + wk * np.array([[1.0], [1.0]])
    x_obs_out = np.vstack((x_obs_out, x_obs))

x_pred_out = np.array([1, 1])
p_pred_out = []

for i in range(10):
    # Steps through 10 time steps updating the estimated position
    x_pred, P_pred = predict(x, P, Fk, Q)

    vk = np.array([np.random.normal(0, np.sqrt(variance))])
    z = H @ x_obs_out[i] + vk*np.array([[1.0]])

    x, P = update_step(x_pred, P_pred, H, z, R, K)
    x_pred_out = np.vstack((x_pred_out, x.T))
    p_pred_out.append([P[0, 0], P[0, 1], P[1, 0], P[1, 1]])


# Plots the output of the kalman filter wrt the actual trajectory
time_plt = np.arange(0, 1, 0.1)
plt.plot(time_plt, x_actual_out[1:, 0], '--', label='x - true')
plt.plot(time_plt, x_actual_out[1:, 1], '--', label='y - true')
plt.plot(time_plt, x_pred_out[1:, 1], label='y - kalman')
plt.plot(time_plt, x_pred_out[1:, 0], label='x - kalman')
plt.xlabel("time")
plt.ylabel("x")
plt.legend()
plt.show()

"""
p_pred_out = np.array(p_pred_out)
fig, axs = plt.subplots(4)
fig.suptitle('State Prediction')
axs[0].plot(time_plt, p_pred_out[:, 0])
axs[1].plot(time_plt, p_pred_out[:, 1])
axs[2].plot(time_plt, p_pred_out[:, 2])
axs[3].plot(time_plt, p_pred_out[:, 3])
plt.xlabel("time")
plt.legend()
plt.show()
"""
