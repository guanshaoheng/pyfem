from utils.mesh import *
from utils.display import *
from TwoDimensionalUtils.TriShapeFunction import *
import numpy as np

'''
        Author: Wang Jingzhou
        Email: Andrewwang@whu.edu.cn
        Affiliation: School of water resources and hydropower engineering, Wuhan University
'''

def setPressure(pressure, mesh, index, coord_value, x, y, f):
    boundary_points = mesh.getBoundaryPoints(index, coord_value)
    boundary_lines = mesh.getBoundaryLines(boundary_points)
    s = sym.symbols("s")
    N1 = (1 - s) / 2
    N2 = (1 + s) / 2
    N = sym.Matrix([N1, N2]).T
    for line in boundary_lines:
        point_1 = mesh.point[line[0]]
        point_2 = mesh.point[line[1]]
        length = point_1.distance(point_2)
        base = np.array([point_1.coord, point_2.coord])
        temp = np.sqrt(1 / 3)
        gaussian_point1 = np.dot(np.array(N.subs([(s, -temp)])), base)
        gaussian_point2 = np.dot(np.array(N.subs([(s, temp)])), base)
        pressure1 = pressure.subs([(x, gaussian_point1[0, 0]), (y, gaussian_point1[0, 1])])
        pressure2 = pressure.subs([(x, gaussian_point2[0, 0]), (y, gaussian_point2[0, 1])])
        pressure_base = np.array([pressure1, pressure2])
        f1 = np.dot(np.array(N.subs([(s, -temp)])), pressure_base) * length / 2
        f2 = np.dot(np.array(N.subs([(s, temp)])), pressure_base) * length / 2
        f[line[0] - 1, index] += f1
        f[line[1] - 1, index] += f2

if __name__ == "__main__":
    # figs save path
    figDirectory = 'biaCompression'
    if not os.path.exists(figDirectory):
        os.mkdir(figDirectory)

    # create the model of calculation
    points = [Point(1, [0, 0]),
              Point(2, [0.3, 0]),
              Point(3, [0.3, 0.7]),
              Point(4, [0, 0.7])]
    lines = [Line(i, [points[i], points[(i + 1) % len(points)]])
             for i in range(len(points))]
    domain = Domain(lines)
    domain.divideBoundary("length", 0.1)
    tri_mesh = triMesh(domain)
    displayTriMesh(tri_mesh)

    # set the force and the displacement to calculate
    num = len(tri_mesh.point_name)
    f = np.zeros((num, 2))
    mask = np.full((num, 2), np.nan)

    # TODO pressure should be calculated with regard to the length of the boundary (2D) (or area of the surface in 3D)
    # TODO add a function to integrate the pressure on the top
    # TODO use the quad mesh
    # TODO save the figures and add the computational results to the readme file

    # set the computational parameter
    x, y = sym.symbols("x, y")
    pressure = 1e6 * x ** 0
    displacement = -0.14
    n_step = 10
    d_displacement_list = [displacement * i / n_step
                           for i in range(n_step + 1)]

    # apply the pressure
    setPressure(pressure, tri_mesh, 0, 0, x, y, f)
    setPressure(-pressure, tri_mesh, 0, 0.3, x, y, f)

    # get the boundary
    points_top = tri_mesh.getBoundaryPoints(1, 0.7)
    points_bottom = tri_mesh.getBoundaryPoints(1, 0)
    for point_name in points_bottom:
        mask[point_name - 1] = np.array([0, 0])
    F_top = []
    for i, d_displacement in enumerate(d_displacement_list):
        for point_name in points_top:
            mask[point_name - 1] = np.array([0, d_displacement])
        # solve
        f_node = twoDimFEM(tri_mesh, f, mask, step=i, filePath=figDirectory)
        F = 0
        for point_name in points_top:
            F += f_node[point_name - 1, 1]
        F_top.append(F)

    plt.figure()
    plt.xlabel("displacement/(m)")
    plt.ylabel("force/(kN)")
    plt.plot((-np.array(d_displacement_list)).tolist(),
             (-np.array(F_top)/1000).tolist())
    fileName = os.path.join(figDirectory, 'force_to_displacement_curve')
    plt.savefig(fileName)

    plt.figure()
    plt.xlabel("strain/(%)")
    plt.ylabel("stress/(MPa)")
    plt.plot((-np.array(d_displacement_list) / 0.7).tolist(),
             (-np.array(F_top) / 1000000 / 0.3).tolist())
    fileName = os.path.join(figDirectory, 'stress_to_strain_curve')
    plt.savefig(fileName)