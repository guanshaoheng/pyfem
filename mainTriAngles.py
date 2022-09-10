from utils.mesh import *
from utils.display import *
from TwoDimensionalUtils.TriShapeFunction import *
import numpy as np

'''
        Author: Wang Jingzhou
        Email: Andrewwang@whu.edu.cn
        Affiliation: School of water resources and hydropower engineering, Wuhan University
'''

if __name__ == "__main__":
    # create the model of calculation
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
    domain.divideBoundary("length", 20)
    mesh = triMesh(domain)
    displayTriMesh(mesh)

    # set the force and the displacement to calculate
    num = len(mesh.point_name)
    f = np.zeros((num, 2))
    # f[2] = f[3] = f[6] = np.array([0, -5e6])qq
    f[0] = f[5] = np.array([5e7, 0])
    mask = np.full((num, 2), np.nan) # node constraints of displacement
    mask[0] = np.array([0, 0])
    mask[1] = np.array([10, 0])
    # mask[4] = np.array([0, 1])
    twoDimFEM(mesh, f, mask, step=0, filePath='./')

