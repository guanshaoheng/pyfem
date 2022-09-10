from TwoDimensionalUtils.QuadShapeFunction import *
from utils.mesh import *
from utils.display import *

'''
        Author: Wang Jingzhou
        Email: Andrewwang@whu.edu.cn
        Affiliation: School of water resources and hydropower engineering, Wuhan University
'''

def setPressurreNew(node, boundary, pressure, index, x, y, f):
    m = node.shape[1]
    s = sym.symbols("s")
    N1 = (1 - s) / 2
    N2 = (1 + s) / 2
    N = sym.Matrix([N1, N2]).T
    for i in range(len(boundary) - 1):
        point1 = node[boundary[i][0], boundary[i][1]]
        point2 = node[boundary[i+1][0], boundary[i+1][1]]
        vector = point1 - point2
        length = np.sqrt(np.dot(vector, vector))
        base = np.array([point1, point2])
        temp = np.sqrt(1 / 3)
        gaussian_point1 = np.dot(np.array(N.subs([(s, -temp)])), base)
        gaussian_point2 = np.dot(np.array(N.subs([(s, temp)])), base)
        pressure1 = pressure.subs([(x, gaussian_point1[0, 0]), (y, gaussian_point1[0, 1])])
        pressure2 = pressure.subs([(x, gaussian_point2[0, 0]), (y, gaussian_point2[0, 1])])
        pressure_base = np.array([pressure1, pressure2])
        f1 = np.dot(np.array(N.subs([(s, -temp)])), pressure_base) * length / 2
        f2 = np.dot(np.array(N.subs([(s, temp)])), pressure_base) * length / 2
        name1 = boundary[i][0] * m + boundary[i][1]
        name2 = boundary[i+1][0] * m + boundary[i+1][1]
        f[name1, index] += f1
        f[name2, index] += f2


if __name__ == "__main__":
    # figs save path
    figDirectory = 'biaCompression'
    if not os.path.exists(figDirectory):
        os.mkdir(figDirectory)

    # create the model of calculation
    points = [[0, 0],
              [0.3, 0],
              [0.3, 0.7],
              [0, 0.7]]
    m = 12
    n = 28
    node, elem = QuadMesh(points, m, n)
    displayQuadMesh(node, elem)

    # set the force and the displacement to calculate
    num = node.shape[0] * node.shape[1]
    f = np.zeros((num, 2))
    mask = np.full((num, 2), np.nan)

    # set the computational parameter
    x, y = sym.symbols("x, y")
    pressure = 1e6 * x ** 0
    displacement = -0.14
    n_step = 10
    d_displacement_list = [displacement * i / n_step
                           for i in range(n_step + 1)]

    # get the boundary
    points_top = [[n, i] for i in range(m + 1)]
    points_bottom = [[0, i] for i in range(m + 1)]
    points_right = [[j, m] for j in range(n + 1)]
    point_left = [[j, 0] for j in range(n + 1)]
    # apply the pressure
    setPressurreNew(node, point_left, pressure, 0, x, y, f)
    setPressurreNew(node, points_right, -pressure, 0, x, y, f)

    for point_i in points_bottom:
        point_name = point_i[0] * (m + 1) + point_i[1]
        mask[point_name] = np.array([0, 0])
    F_top = []
    for i, d_displacement in enumerate(d_displacement_list):
        for point_i in points_top:
            point_name = point_i[0] * (m + 1) + point_i[1]
            mask[point_name] = np.array([0, d_displacement])
        # solve
        f_node = twoDimFEM(node, elem, f, mask, step=i, filePath=figDirectory)
        F = 0
        for point_i in points_top:
            point_name = point_i[0] * (m + 1) + point_i[1]
            F += f_node[point_name, 1]
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