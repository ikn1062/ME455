import numpy as np
import matplotlib.pyplot as plt
import copy
import random

np.set_printoptions(linewidth=600)


def initialize_door(map, y, x):
    n = np.size(map)
    map[y][x] = 1

    y_list_1 = [y - 1, y + 1]
    for i in y_list_1:
        for j in range(-1, 2):
            if n > i >= 0 and 0 <= (x + j) < n:
                map[i][x + j] = 0.5

    y_list_1 = [y - 2, y + 2]
    for i in y_list_1:
        for j in range(-2, 3):
            if n > i >= 0 and 0 <= (x + j) < n:
                map[i][x + j] = 1 / 3

    y_list_1 = [y - 3, y + 3]
    for i in y_list_1:
        for j in range(-3, 4):
            if n > i >= 0 and 0 <= (x + j) < n:
                map[i][x + j] = 1 / 4

    return map


def calc_entropy(map):
    S = 0
    for y in map:
        for p in y:
            if p < 0.00000001:
                p = 0.00000001
            S -= p * np.log(p)
    return S


def get_Lr0(map, y, x):
    y_list, x_list = [y + 1, y - 1], [x + 1, x - 1]
    p_list = []
    n = len(map[0])

    for i in y_list:
        if 0 <= i < n:
            p_list.append(map[i][x])
    for j in x_list:
        if 0 <= j < n:
            p_list.append(map[y][j])

    Lr0 = sum(p_list) / len(p_list)
    return Lr0


def update_likelihood(z, likelihood, y, x):
    n = len(likelihood[0])

    if z == 1:
        likelihood[y][x] = 1
        y_list_1 = [y - 1, y + 1]
        for i in y_list_1:
            for j in range(-1, 2):
                if n > i >= 0 and 0 <= (x + j) < n and likelihood[i][x + j] > 0.001:
                    likelihood[i][x + j] = 0.5

        y_list_1 = [y - 2, y + 2]
        for i in y_list_1:
            for j in range(-2, 3):
                if n > i >= 0 and 0 <= (x + j) < n and likelihood[i][x + j] > 0.001:
                    likelihood[i][x + j] = 1 / 3

        y_list_1 = [y - 3, y + 3]
        for i in y_list_1:
            for j in range(-3, 4):
                if n > i >= 0 and 0 <= (x + j) < n and likelihood[i][x + j] > 0.001:
                    likelihood[i][x + j] = 1 / 4

    else:
        likelihood[y][x] = 0.000001
        y_list_1 = [y - 1, y + 1]
        for i in y_list_1:
            for j in range(-1, 2):
                if n > i >= 0 and 0 <= (x + j) < n and likelihood[i][x + j] > 0.001:
                    likelihood[i][x + j] = 0.5

        y_list_1 = [y - 2, y + 2]
        for i in y_list_1:
            for j in range(-2, 3):
                if n > i >= 0 and 0 <= (x + j) < n and likelihood[i][x + j] > 0.001:
                    likelihood[i][x + j] = 2 / 3

        y_list_1 = [y - 3, y + 3]
        for i in y_list_1:
            for j in range(-3, 4):
                if n > i >= 0 and 0 <= (x + j) < n and likelihood[i][x + j] > 0.001:
                    likelihood[i][x + j] = 3 / 4

    return likelihood


def update_prob(z, prior_prob, visited, likelihood):
    n = len(visited[0])
    prob = np.copy(prior_prob)
    px1 = 1 / (n ** 2)
    if z == 1.0:
        px = px1
    else:
        px = 1 - px1
    for i in range(n):
        for j in range(n):
            Lr0 = get_Lr0(likelihood, i, j)
            pr0 = visited[i][j]
            prob[i][j] = Lr0 * pr0 / px

    prob = prob / np.sum(prob)
    return prob


def update_visited(visited, y, x, n_unvisited):
    n = len(visited[0])
    visited[y][x] = 0.0000
    for i in range(n):
        for j in range(n):
            if visited[i][j] > 0.00001:
                visited[i][j] = 1 / n_unvisited

    return visited


def calc_expS(y, x, prob, visited, n_unvisited_, measurement_l, S):
    n = len(prob[0])
    p1 = measurement_l[y][x]
    p0 = 1 - p1
    likelihood_ = 0.01 * np.ones((n, n))
    likelihood_0, likelihood_1 = update_likelihood(0, likelihood_, y, x), update_likelihood(1, likelihood_, y, x)
    visited = update_visited(visited, y, x, n_unvisited_ - 1)
    prob_0, prob_1 = update_prob(0, prob, visited, likelihood_0), update_prob(1, prob, visited, likelihood_1)
    S0, S1 = calc_entropy(prob_0), calc_entropy(prob_1)
    Es = p0 * (S0 - S) + p1 * (S1 - S)
    return Es


def calc_control(z, y, x, prob, visited, n_unvisited, S_curr, measurement_l):
    visited_ = copy.deepcopy(visited)
    n_unvisited_ = n_unvisited
    n = len(prob[0])
    u_options = [[1, 0], [0, -1], [-1, 0], [0, 1], [0, 0]]
    exp_entropy = 1000 * np.ones(5)
    i = 0
    for u in u_options:
        y1 = y + u[0]
        x1 = x + u[1]
        if y1 >= 0 and y1 < n and x1 >= 0 and x1 < n:
            S_new = calc_expS(y1, x1, prob, visited_, n_unvisited_, measurement_l, S_curr)
            delS = (1 - prob[y1][x1]) * S_new - ((prob[y1][x1]) * S_curr)
            exp_entropy[i] = delS
        i += 1

    print(exp_entropy)
    min_index = np.argmin(exp_entropy)

    return u_options[min_index]


def loop():
    n = 25
    likelihood = 0.01 * np.ones((n, n))
    x_door, y_door = 4, 13
    likelihood = initialize_door(likelihood, y_door, x_door)
    visited = (1 / (n ** 2)) * np.ones((n, n))
    map = np.zeros((n, n))
    map[y_door][x_door] = 1.0

    x_start, y_start = 21, 17
    prior_prob = (1 / (n ** 2)) * np.ones((n, n))
    H = calc_entropy(prior_prob)
    n_unvisited = n ** 2
    visited_list = [[y_start, x_start]]

    epsilon = 0.0000015
    x_next, y_next = x_start, y_start
    path_x, path_y = [x_start], [y_start]

    while H > epsilon and n_unvisited > 1:
        if x_door != x_start or y_door != y_start:
            x_start, y_start = x_next, y_next

            plt.scatter(y_start, x_start)
            plt.draw()
            plt.pause(0.01)

            l = likelihood[y_start][x_start]
            z = random.choices([0, 1], weights=[1 - l, l])[0]

            ctr = 0
            for v in visited_list:
                if v[0] == y_start and v[1] == x_start:
                    ctr = 1
            if ctr == 0:
                visited_list.append([y_start, x_start])

            measurement_l = 0.01 * np.ones((n, n))
            measurement_l = update_likelihood(z, measurement_l, y_start, x_start)
            n_unvisited = n ** 2 - (len(visited_list))

            visited = update_visited(visited, y_start, x_start, n_unvisited)

            post_prob = update_prob(z, y_start, x_start, prior_prob, visited, n_unvisited, measurement_l)
            H = calc_entropy(post_prob)
            u = calc_control(z, y_start, x_start, post_prob, visited, n_unvisited, H, measurement_l)

            prior_prob = copy.deepcopy(post_prob)
            y_next, x_next = y_start + u[0], x_start + u[1]
            plt.imshow(post_prob.T, origin="lower")
            path_x.append(x_start)
            path_y.append(y_start)
        else:
            break

    plt.figure()
    plt.plot(path_y, path_x)
    plt.scatter(path_y, path_x)
    plt.scatter(y_door, x_door, marker='x')
    plt.scatter(path_y[0], path_x[0], marker='o')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.xlim(0, 25)
    plt.ylim(0, 25)
    plt.show()
    plt.savefig('path.png')


if __name__ == "__main__":
    loop()
