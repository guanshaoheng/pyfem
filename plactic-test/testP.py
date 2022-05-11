# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from mesh import QuadMesh
from display import displayQuadMesh
from Constitutive import VonMises
from ElementType import Q4
from Solver import Solver

if __name__ == "__main__":
    # 设置问题的维度
    n_dim = 2

    # create the model of calculation
    points = [[0, 0],
              [0.3, 0],
              [0.3, 0.7],
              [0, 0.7]]
    m = 3
    n = 7
    node, elem = QuadMesh(points, m, n)
    displayQuadMesh(node, elem)

    # 本构和单元类型设置
    E, mu = 1e8, 0.2
    sigma_s = 1e7
    Et = 1e7
    n_gauss = 4
    elements = []
    for i in range(np.shape(elem)[0]):
        cons = VonMises(n_gauss, n_dim, E, mu, sigma_s, Et)
        x = np.array([node[j] for j in elem[i]])
        node_list = elem[i]
        elements.append(Q4(node_list, x, cons))

    # 获取边界点集
    points_top = [i for i in range(len(node))
                  if abs(node[i, 1] - 0.7) <= 1e-5]
    points_bottom = [i for i in range(len(node))
                     if abs(node[i, 1]) <= 1e-4]
    points_right = [i for i in range(len(node))
                    if abs(node[i, 0] - 0.3) <= 1e-5]
    point_left = [i for i in range(len(node))
                  if abs(node[i, 0]) <= 1e-5]

    # 边界条件数值设置
    x, y = sym.symbols("x, y")
    coord_base = [x, y]
    pressure = 1e6 * x ** 0
    displacement = -0.14
    n_step = 25

    # 实例化求解器
    solver = Solver(n_dim, node, elements)

    # 设置围压
    solver.setPressure(point_left, pressure, coord_base)
    solver.setPressure(points_right, -pressure, coord_base)

    # 设置位移边界条件及按位移加载
    solver.setDisplacement(points_bottom, 0, 0)
    solver.setDisplacement(points_bottom, 1, 0)
    solver.setDisplacement(points_top, 1, displacement)
    solver.split("displacement", n_step)

    # 求解
    u, f_node, V = solver.solve()

    # 计算顶部位移及反力大小
    u_top = [displacement * i / n_step
             for i in range(n_step + 1)]
    F = solver.integralF(points_top)
    F_top = [F[i][1]
             for i in range(n_step + 1)]

    D = np.array(V) / (0.7 + np.array(u_top))
    stress = -np.array(F_top) / D / 1e6
    strain_l = -np.array(u_top) / 0.7 * 100
    strain_v = -(np.array(V) - V[0]) / V[0] * 100

    # 绘制应力应变曲线
    plt.figure()
    plt.xlabel("strain_l/(%)")
    plt.ylabel("stress/(MPa)")
    plt.plot(strain_l, stress, "o-")
    plt.show()

    # 绘制体积应变曲线
    plt.figure()
    plt.xlabel("strain_l/(%)")
    plt.ylabel("strain_v/(%)")
    plt.plot(strain_l, strain_v, "o-")
    plt.show()