from mesh import *
from display import *
from solve import *
import numpy as np

if __name__ == "__main__":
    points = [Point(1, [-5, 0]),
              Point(2, [100, 0]),
              Point(3, [20, 280]),
              Point(4, [20, 300]),
              Point(5, [0, 300]),
              Point(6, [0, 150])]
    '''points = [Point(1, [0, 0]),
              Point(2, [2, 0]),
              Point(3, [2, 2]),
              Point(4, [0, 2])]'''
    '''points = [Point(1, [1, 0]),
              Point(2, [2, 1]),
              Point(3, [0, 3]),
              Point(4, [-2, 1]),
              Point(5, [-1, 0]),
              Point(6, [-2, -1]),
              Point(7, [0, -3]),
              Point(8, [2, -1])]'''
    lines = [Line(i, [points[i], points[(i + 1) % len(points)]])
             for i in range(len(points))]
    domain = Domain(lines)
    # domain.divideBoundary("length", 20)
    mesh = mesh(domain)
    displayMesh(mesh)
    num = len(mesh.point_name)
    f = np.zeros((num, 2))
    # f[2] = f[3] = f[6] = np.array([0, -5e6])qq
    f[0] = f[5] = np.array([5e7, 0])
    mask = np.zeros((num, 2))  # node constraints of displacement
    mask[0] = np.array([1, 1])
    mask[1] = np.array([0, 1])
    # mask[4] = np.array([0, 1])
    twoDimFEM(mesh, f, mask)

