import numpy as np
import matplotlib.pyplot as plt


def dynamcis(x, u):
    """
    Computes x_dot of system given position x and control u
    """
    xdot = np.cos(x[2]) * u[0]
    ydot = np.sin(x[2]) * u[0]
    tdot = u[1]
    return np.array([xdot, ydot, tdot])


def integrate(f, xt, dt, u):
    """
    Runge-Kutta 4th order integral method, given function f
    """
    k1 = dt * f(xt, u)
    k2 = dt * f(xt + k1/2, u)
    k3 = dt * f(xt + k2/2, u)
    k4 = dt * f(xt + k3, u)
    x_next = xt + (1/6 * (k1 + 2*k2 + 2*k3 + k4))
    return x_next


def initial_traj(f, x0, timeline, dt, integrate, u):
    """
    Creates initial trajectory given dynamics f, initial position, control, and time scaling
    """
    N = int((max(timeline) - min(timeline))/dt)
    x = np.copy(x0)
    x_traj = np.zeros((len(x0), N))

    for i in range(N):
        x_traj[:, i] = integrate(f, x, dt, u[:, i])
        x = np.copy(x_traj[:, i])

    return x_traj


def obj(args, Q, R, P1):
    """
    Calculates objective function given position x, control u, and system parameters
    """
    J = np.zeros((N))
    t = 0
    for i in range(N):
        x = np.array([args[0:3, i]])
        xd = np.array([2*t/np.pi, 0, np.pi/2])
        u = np.array([args[3:5, i]])
        err = x - xd

        err_Q_errT = np.matmul(err, np.matmul(Q, np.transpose(err)))
        u_R_uT = np.matmul(u, np.matmul(R, np.transpose(u)))
        J_val = 0.5 * (err_Q_errT + u_R_uT)

        J[i] = J_val[0][0]
        t += dt

    xT = np.array([args[0:3, -1]])
    xdT = np.array([4, 0, np.pi / 2])
    xdT_err = xT - xdT

    J_val = 0.5 * np.matmul(xdT_err, np.matmul(P1, np.transpose(xdT_err)))
    trapz = np.trapz(J, dx=dt) + J_val[0][0]

    return trapz


def calc_P(args, Q, R, P1):
    """
    Calculates control parameter P used to solve iLQR
    """
    t = np.arange(0, T, dt)

    list_P = np.zeros((t.size, 3, 3))
    list_r = np.zeros((t.size, 3, 1))

    Rinv = np.linalg.inv(R)
    P = P1.copy()

    r = -np.array([4, 0, np.pi/2]).T
    list_r[0] = np.transpose([r])
    list_P[0] = P1.copy()

    time = 0

    for i in range(t.size - 1):
        x = np.array([args[0:3, t.size-2-i]])
        u = np.array([args[3:5, t.size-2-i]])

        A = np.array([[0, 0, -np.sin(x[0][2])*u[0][0]], [0, 0, np.cos(x[0][2])*u[0][0]], [0, 0, 0]])
        B = np.array([[np.cos(x[0][2]), 0], [np.sin(x[0][2]), 0], [0, 1]])

        P_A = np.matmul(P, A)
        P_At = np.matmul(np.transpose(A), P)
        B_Rinv_Bt = np.matmul(B, np.matmul(Rinv, np.transpose(B)))

        P_B_Rinv_Bt_P = np.matmul(P, np.matmul(B_Rinv_Bt, P))

        P_dot = P_A + P_At - P_B_Rinv_Bt_P + Q

        xd = np.array([[2*t[t.size-2-i]/np.pi, 0, np.pi/2]])
        error = x - xd
        a = np.transpose(np.matmul(error, Q))
        b = np.transpose(np.matmul(u, R))
        B_Rinv_Bt_P = np.matmul(B_Rinv_Bt, P)
        P_B_Rinv_b = np.matmul(np.matmul(P, np.matmul(B, Rinv)), b)
        A_1 = np.transpose(A - B_Rinv_Bt_P)

        r_dot = np.matmul(A_1, r) + a.T - P_B_Rinv_b.T

        r += dt * r_dot.flatten()
        P += dt * P_dot
        list_P[i+1] = P
        list_r[i+1] = np.transpose([r])
        time += dt

    return list_P, list_r


def descent_dir(P_list, r_list, args, Q, R, P1, per):
    """
    Finds descent direction given position, control, P, r, system inputs, and scaling based on armijo line search
    """
    t = np.arange(0, T, dt)
    z = per * np.array([[0], [0], [0]])

    Rinv = np.linalg.inv(R)
    zeta = np.zeros((t.size, 5, 1))

    for i in range(t.size - 1):
        r = r_list[i]
        P = P_list[i]
        x = np.array([args[0:3, i]])
        u = np.array([args[3:5, i]])
        b = np.transpose(np.matmul(u, R))
        A = np.array([[0, 0, -np.sin(x[0][2])*u[0][0]], [0, 0, np.cos(x[0][2])*u[0][0]], [0, 0, 0]])
        B = np.array([[np.cos(x[0][2]), 0], [np.sin(x[0][2]), 0], [0, 1]])

        Rinv_BT_P_z = np.matmul(Rinv, np.matmul(np.transpose(B), np.matmul(P, z)))
        Rinv_BT_r = np.matmul(Rinv, np.matmul(B.T, r))

        v = -Rinv_BT_P_z - Rinv_BT_r - np.matmul(Rinv, b)

        zeta[i] = np.vstack((z, v))

        zdot = np.matmul(A, z) + np.matmul(B, v)
        z += zdot * dt

    return zeta


def DJ(zeta, args, Q, R, P1):
    """
    Calculates directional derivative given zeta, position, control, and system inputs
    """
    t = np.arange(0, T, dt)
    J = np.zeros((t.size))

    for i in range(t.size - 1):
        z = zeta[i, :3]
        xd = np.array([[2*t[i]/np.pi, 0, np.pi/2]])
        x = np.array([args[0:3, i]])
        err = x - xd
        u = np.array([args[3:5, i]])
        v = zeta[i, 3:]
        err_Q_z = np.matmul(err, np.matmul(Q, z))
        u_R_v = np.matmul(u, np.matmul(R, v))
        J_val = err_Q_z + u_R_v
        J[i] = J_val[0][0]

    trapz = np.trapz(J, dx=dt)
    return trapz


if __name__ == "__main__":
    # Sets up intial parameters for system
    T = 2*np.pi
    timeline = [0, T]
    dt = 0.001
    N = int((max(timeline) - min(timeline))/dt)
    u_init = np.array([1, -1/2])
    x0 = np.array([0, 0, np.pi/2])
    Q = 1.0 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    R = 1.0 * np.array([[1, 0], [0, 1]])
    P1 = 1.0 * np.array([[35, 0, 0], [0, 120, 0], [0, 0, 5]])

    # Creates initial trajectory as well as control vector
    u = np.vstack((np.ones(N), (-1/2) * np.ones(N)))
    traj_initial = initial_traj(dynamcis, x0, timeline, dt, integrate, u)
    traj_initial = np.vstack((traj_initial, u))


    # Sets up variables for gradient descent
    traj = traj_initial
    alpha = 0.40
    beta = 0.85
    eps = 0.01
    i = 0
    per = 0.0

    # Plot initial trajectory of system
    plt.figure()
    plt.title("Initial Trajectory")
    plt.plot(traj[0, :], traj[1, :])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


    while True:
        # Creates P and r vectors for calculation
        P_list, r_list = calc_P(traj, Q, R, P1)
        P_list = np.flip(P_list, 0)
        r_list = np.flip(r_list, 0)
        zeta = descent_dir(P_list, r_list, traj, Q, R, P1, per)

        # Resets gamma out of loop
        n = 0
        gamma = beta**n

        # Creates new trajectory to optimize
        traj_a = traj.copy()

        while n < 30:
            # Finds a new trajectory with armijo line search
            x = np.array([traj[0:3, :]])
            u = np.array([traj[3:5, :]])
            v = np.transpose(zeta[:, 3:5])
            u_p = u + gamma*v[:, :, :-1]
            u_p = u_p[0]
            x0 = x[0, :, 0]

            traj_a[0:3, :] = initial_traj(dynamcis, x0, timeline, dt, integrate, u_p)
            traj_a[3:5, :] = u_p

            obj_traj_a = obj(traj_a, Q, R, P1)
            obj_traj = obj(traj, Q, R, P1)
            dj = DJ(zeta, traj, Q, R, P1)

            if obj_traj_a > (obj_traj + alpha*gamma*dj):
                n += 1
                gamma = beta**n
            else:
                break

        traj[0:3] = traj_a[0:3]
        traj[3:5] = traj_a[3:5]
        i += 1

        # If DJ of system converges, end while loop
        if abs(DJ(zeta, traj, Q, R, P1)) < eps:
            break

        # Plots of current trajectory and control signal in loop
        plt.figure()
        plt.title("Current Trajectory")
        plt.plot(traj[0, :], traj[1, :])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        fig, axs = plt.subplots(2, 1)
        t_plot = np.arange(0, T, dt)
        plt.title("Current Control Signal")
        axs[0].plot(t_plot[:-1], traj[3, :])
        axs[1].plot(t_plot[:-1], traj[4, :])
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('u0')
        axs[1].set_ylabel('u1')
        fig.tight_layout()
        plt.show()




