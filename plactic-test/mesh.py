import numpy as np

def reflect(points, i, j, m, n):
    xi = 2 / m * i - 1
    eta = 2 / n * j - 1
    N1 = (1 - xi) * (1 - eta) / 4
    N2 = (1 + xi) * (1 - eta) / 4
    N3 = (1 + xi) * (1 + eta) / 4
    N4 = (1 - xi) * (1 + eta) / 4
    N = np.array([N1, N2, N3, N4])
    coord = np.dot(N, points)
    return coord.tolist()

def QuadMesh(ori_points, m, n):
    points = np.array(ori_points)
    node = []
    elem = []
    for i in range(m + 1):
        node.append(reflect(points, i, 0, m, n))
    for j in range(1, n + 1):
        node.append(reflect(points, 0, j, m, n))
        for i in range(1, m + 1):
            node.append(reflect(points, i, j, m, n))
            elem.append([(j - 1) * (m + 1) + i - 1,
                         (j - 1) * (m + 1) + i,
                         j * (m + 1) + i,
                         j * (m + 1) + i - 1])
    return np.array(node), np.array(elem)