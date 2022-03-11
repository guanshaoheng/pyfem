from TwoDimensionalUtils.mesh import *
from TwoDimensionalUtils.display import *
from TwoDimensionalUtils.TriShapeFunction import *
import numpy as np

'''
        Author: Wang Jingzhou
        Email: Andrewwang@whu.edu.cn
        Affiliation: School of water resources and hydropower engineering, Wuhan University
'''

if __name__ == "__main__":
    # create the model of calculation
    points = [Point(1, [0, 0]),
              Point(2, [0.3, 0]),
              Point(3, [0.3, 0.7]),
              Point(4, [0, 0.7])]
    lines = [Line(i, [points[i], points[(i + 1) % len(points)]])
             for i in range(len(points))]
    domain = Domain(lines)
    # for i in [0.1, 0.2]:
    #     for j in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    #         new_point_name = max(domain.point_name) + 1
    #         domain.addPoint(Point(new_point_name, [i, j]))
    domain.divideBoundary("length", 0.03)
    mesh = mesh(domain)
    displayMesh(mesh)

    # set the force and the displacement to calculate
    num = len(mesh.point_name)
    f = np.zeros((num, 2))
    mask = np.full((num, 2), np.nan)
    # apply pressure
    pressure = 1.0e6
    points_left = mesh.getBoundaryPoints(0, 0)
    for point_name in points_left:
        f[point_name - 1, 0] = pressure
    points_right = mesh.getBoundaryPoints(0, 0.3)
    for point_name in points_right:
        f[point_name - 1, 0] = -pressure
    # apply displacement
    for i in range(10):
        displacement = -0.014 * (i + 1)
        points_bottom = mesh.getBoundaryPoints(1, 0)
        for point_name in points_bottom:
            mask[point_name - 1] = np.array([0, 0])
        points_top = mesh.getBoundaryPoints(1, 0.7)
        for point_name in points_top:
            mask[point_name - 1] = np.array([0, displacement])
        twoDimFEM(mesh, f, mask)