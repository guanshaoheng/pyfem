import matplotlib.pyplot as plt
import numpy as np
from utils.ShapeFunction import TwoDimensionShape, plotElement, \
    stiffnessAssembling, plotGuassian, saveVTK

# import pyvista as pv
# import vtk

if __name__ == '__main__':
    ndim = 2
    Nnum = 4

    # -------------------------------------------------------------------
    # configuration 1 (node 11 element 6)
    nodeCoord = np.array([[0, 0], [1, 0], [1, 1],
                          [0, 1], [2, 0], [2, 1],
                          [2, 2], [1, 2], [0, 2],
                          [2, 3], [1, 3], [0, 3]], dtype=np.float)
    node2Element = np.array([[0, 1, 2, 3], [1, 4, 5, 2], [2, 5, 6, 7], [3,2,7,8],
                             [7, 6, 9, 10], [8, 7, 10, 11]])
    f = np.zeros_like(nodeCoord, dtype=np.float)  # node force
    # f[9] =f[10]=f[11]= np.array([2e6, 0])
    f[9] =f[10]=f[11]= np.array([0., -5e6])
    mask = np.zeros_like(nodeCoord)  # node constraints of displacement
    mask[0] = np.array([0, 1])
    mask[4] = np.array([0, 1])
    mask[1] = np.array([1, 1])

    # configuration 2 (node 9 element 4)
    # nodeCoord = np.array([[0, 0], [1, 0], [2, 0], [0, 1],
    #                       [1, 1], [2, 1], [0, 2], [1, 2],
    #                       [2, 2]], dtype=np.float)
    # node2Element = np.array([[1, 4, 3, 0], [2, 5, 4, 1], [5, 8, 7, 4], [7, 6, 3, 4]])
    # f = np.zeros_like(nodeCoord, dtype=np.float)  # node force
    # f[2] = f[5] = f[8] = np.array([0, 2e6])
    # mask = np.zeros_like(nodeCoord)  # node constraints of displacement
    # mask[0] = np.array([1, 1])
    # mask[2] = np.array([0, 1])

    # configuration 3 (node 8 element 3)
    # nodeCoord = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1], [3, 0], [3, 1]], dtype=np.float)
    # node2Element = np.array([[0, 1, 2, 3], [1, 4, 5, 2], [4,6,7,5]])
    # f = np.zeros_like(nodeCoord, dtype=np.float)  # node force
    # f[2] = f[5] = np.array([0, -2e6])
    # mask = np.zeros_like(nodeCoord)  # node constraints of displacement
    # mask[0] = np.array([1, 1])
    # mask[6] = np.array([0, 1])

    # configuration 4 (node 6 element 2)
    # nodeCoord = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]], dtype=np.float)
    # node2Element = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
    # f = np.zeros_like(nodeCoord, dtype=np.float)  # node force
    # f[2] = np.array([-5e6, -5e6])
    # mask = np.zeros_like(nodeCoord)  # node constraints of displacement
    # mask[0] = np.array([1, 1])
    # mask[4] = np.array([0, 1])

    nodeNum, elementNum = len(nodeCoord), len(node2Element)
    nodeIndex = np.arange(nodeNum)
    elementIndex = np.arange(elementNum)

    domain = TwoDimensionShape()
    k = []
    for i in range(elementNum):
        k.append(domain.getElementStiffness(x=nodeCoord[node2Element[i]]))
    K_global = stiffnessAssembling(nodeIndex=nodeIndex, kList=k, node2Element=node2Element)

    uValue = mask * 0.
    k_free, f_free = domain.displacementBoundaryCondition(K_global=K_global, mask=mask, f=f)
    u, f_node = domain.solve(mask=mask, K_global=K_global, K_free=k_free, f_free=f_free, uValue=uValue)
    epsilon, gaussianCoord = domain.getStrain(u=u, node2Element=node2Element, nodeCoord=nodeCoord+u)
    epsilonNode = domain.interpolateStrainToNode(nodeCoord, node2Element, epsilon)

    x_ = nodeCoord + u
    plotGuassian(gaussianCoord, 'ro')
    for i in range(elementNum):
        plotElement(nodeCoord[node2Element[i]], 'ko-')
        plotElement(x_[node2Element[i]], 'bo-')
    for j in range(nodeNum):
        plt.text(x=nodeCoord[j, 0], y=nodeCoord[j, 1], s=str(j))
    plt.axis('equal')
    plt.show()
    # plt.savefig("./node_%d_element_%d.png" % (nodeNum, elementNum), pmi=200)

    # ---------------------------------------------
    # write to vtk (customization)
    saveVTK('./results.vtk', x_, node2Element, **{"u": u, "$\epsilon$": epsilonNode})

    # ------------------------------------------------------
    # # pyvista
    # nodeCoord_3 = np.concatenate((nodeCoord+u, np.zeros(shape=(len(nodeCoord), 1), dtype=np.float)), axis=1)
    # grid = pv.UnstructuredGrid({vtk.VTK_QUAD: node2Element}, nodeCoord_3)
    # simple_range = range(grid.n_cells)
    # simple_list = list(range(grid.n_points))
    # # grid.cell_data['my-data'] = simple_range
    # grid.point_data['$u_x$'] = u[:, 0]
    # grid.point_data['$u_y$'] = u[:, 1]
    # grid.point_data['$\epsilon_{xx}$'] = epsilonNode[:, 0, 0]
    # grid.point_data['$\epsilon_{yy}$'] = epsilonNode[:, 1, 1]
    # grid.point_data['$\epsilon_{xy}$'] = epsilonNode[:, 0, 1]
    # grid.plot(show_edges=True, cpos='xy')
    # grid.save('./results_pyvist.vtk', binary=False)

    #
    #     # -----------------------------------------------------
    #     # gaussian points
    #     f.write('\n')
    #     f.write('POINTS %u float\n' % (gaussianCoord.shape[0]*gaussianCoord.shape[1]))
    #     for element in gaussianCoord:
    #         for gaussian in element:
    #             f.write('%f %f %f\n' % (gaussian[0], gaussian[1], 0.))
    #     f.write('\n')
    #
    #     f.write('\nPOINT_DATA %u\n' % (gaussianCoord.shape[0]*gaussianCoord.shape[1]))
    #     f.write("VECTORS strain float\n")
    #     for epsilon_element in epsilon:
    #         for epsilon_gaussian in epsilon_element:
    #             f.write('%f %f %f\n' % (epsilon_gaussian[0, 0], epsilon_gaussian[1, 1], epsilon_gaussian[0, 1]))
