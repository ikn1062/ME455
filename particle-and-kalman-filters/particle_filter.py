import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from scipy import stats

"""
Code heavily modified from: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
"""

def update_pos(x, u, dt):
    """
    Updates the position based on the defined dynamics
    """
    x_next = x[0] + (np.cos(x[2]) * u[0] * dt)
    y_next = x[1] + (np.sin(x[2]) * u[0] * dt)
    t_next = x[2] + (u[1] * dt)
    return np.array([x_next, y_next, t_next])


def create_particles(mean, std, N):
    """
    Creates an initial set of particles for the filter
    """
    particles = np.empty((N, 3))
    for i in range(3):
        particles[:, i] = mean[i] + (np.random.randn(N) * std[i])
    particles[:, 2] %= 2 * np.pi
    return particles


def predict(particles, u, std, dt=0.1):
    """
    Predicts the output of the particles given the control input, dt, and std with noise
    """
    N = len(particles)

    # Update the angle of the object with noise
    particles[:, 2] += (u[1]*dt) + (np.random.randn(N) * std)
    particles[:, 2] %= 2 * np.pi

    # Move in a direction with noise
    d = (u[0] * dt) + np.random.randn(N) * std
    particles[:, 0] += np.cos(particles[:, 2]) * d
    particles[:, 1] += np.sin(particles[:, 2]) * d


def update(particles, weights, z, std, landmarks):
    """
    Updates the distances based on distances from the landmarks
    """
    for i, lm in enumerate(landmarks):
        dist = np.linalg.norm(particles[:, 0:2] - lm, axis=1)
        weights *= stats.norm(dist, std).pdf(z[i])

    # Normalizes the weights
    weights += 1.e-300
    weights /= sum(weights)


def estimate(particles, weights):
    """
    Estimates the position of the particle based on weights
    """
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean, var


def resample_from_index(particles, weights, indexes):
    """
    Resamples the weights 
    """
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill(1.0 / len(weights))


def systematic_resample(weights):
    """
    Systematic resample of weights before estimated measurement
    """
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N and j < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def neff(weights):
    return 1. / np.sum(np.square(weights))


def particle_filter_trajectory(initial_guess, u, std, N=1500, T=6, dt=0.1, show_particles=True):
    # Generate particles
    particle_std = (3, 3, np.pi / 4)
    particles = create_particles(initial_guess, particle_std, N)

    # Create weights
    weights = np.ones(N) / N

    # Used
    landmarks = np.array([[-2, -2], [0, 5], [4, 3], [5, -2]])
    Nl = len(landmarks)

    robot_pos = initial_guess.copy()

    n = int((T + dt) / dt)
    for _ in range(n):
        # Update current robot pos
        robot_pos = update_pos(robot_pos, u, dt)

        # Calculate distances from robot to landmarks
        zs = np.linalg.norm(landmarks - robot_pos[:2], axis=1) + np.random.randn(Nl) * std

        # Move forward in the next point based on the trajectory
        predict(particles, u, std)

        # Update weights
        update(particles, weights, zs, std, landmarks)

        # Resample weights
        if neff(weights) < N / 2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
            assert np.allclose(weights, 1 / N)

        # Take estimates
        mu, var = estimate(particles, weights)

        # Plot particles, original position, and estimated position
        if show_particles:
            plt.scatter(particles[:, 0], particles[:, 1], color='m', alpha=0.01, marker=',', s=1)
        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+', color='k', s=180, lw=3)
        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')

    plt.legend([p1, p2], ['Initial Trajec', 'Particle Filter'], loc=4, numpoints=1)
    plt.show()

    return 0


if __name__ == "__main__":
    init_pos = np.array([0, 0, np.pi / 2])
    U = np.array([1, -0.5])
    var = 0.02

    particle_filter_trajectory(init_pos, U, np.sqrt(var))

