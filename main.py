import numpy as np
from matplotlib import pyplot as plt
from utils.ShapeFunction import TwoDimensionShape, stiffnessAssembling, plotElement, plotGuassian
from utils.display2Dfield import plot_node_displacement

if __name__ == "__main__":
    # ----------------------------------------------------
    # shape function
    domain = TwoDimensionShape()
    nodeCoord = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=float)
    node2Element = np.array([[0, 1, 2, 3]])
    elementNum = len(node2Element)
    nodeIndex = np.arange(len(nodeCoord))
    x = nodeCoord[node2Element[0]]
    k = []
    for i in range(elementNum):
        k.append(domain.getElementStiffness(x=nodeCoord[node2Element[i]]))
    K_global = stiffnessAssembling(nodeIndex=nodeIndex, kList=k, node2Element=node2Element)

    ndim = x.shape[-1]
    nNode = x.shape[0]

    # f = np.array([[0, 0], [0, 0], [-1e6, -0], [0, 0]])
    f = np.zeros_like(x, dtype=float)
    f[2] = np.array([0., 1e6])

    # ----------------------------------------------------
    # boundary conditions
    mask_constrained = np.zeros_like(x)
    mask_constrained[0] = np.array([1, 1])  # where the 1st boundary condition is applied
    mask_constrained[1] = np.array([0, 1])  # where the 1st boundary condition is applied
    uValue = mask_constrained * 0.
    k_free, f_free = domain.displacementBoundaryCondition(K_global=K_global, mask=mask_constrained, f=f)
    u, f_node = domain.solve(mask=mask_constrained, K_global=K_global,
        K_free=k_free, f_free=f_free, uValue=uValue)

    epsilon, gaussianCoord = domain.getStrain(u=u, node2Element=node2Element, nodeCoord=nodeCoord+u)

    x_ = nodeCoord+u
    plotElement(nodeCoord, 'ro-')
    plotElement(x_, 'bo-')
    plotGuassian(gaussianCoord, 'ro')
    for i in range(4):
        plt.text(x=nodeCoord[i, 0], y=nodeCoord[i, 1], s='%d' % i, fontsize=15)
    # plt.axis('equal')
    # plt.show()

    # ---------------------------------------------
    # plot displacement field
    plot_node_displacement(nodes_x=x_[:, 0], nodes_y=x_[:, 1],
                           nodal_values=u[:, 0], elements=node2Element, title='$u_{x}$')
    plot_node_displacement(nodes_x=x_[:, 0], nodes_y=x_[:, 1],
                           nodal_values=u[:, 1], elements=node2Element, title='$u_{y}$')

    # plot strain
    epsilonNode = domain.interpolateStrainToNode(nodecoord=nodeCoord, node2Elment=node2Element, epsilon=epsilon)
    objectValue = []
    for node in epsilonNode:
        objectValue.append(node[0, 0])
    plot_node_displacement(nodes_x=x_[:, 0], nodes_y=x_[:, 1],
                           nodal_values=objectValue, elements=node2Element, title='$\epsilon_{xx}$')
    objectValue = []
    for node in epsilonNode:
        objectValue.append(node[1, 1])
    plot_node_displacement(nodes_x=x_[:, 0], nodes_y=x_[:, 1],
                           nodal_values=objectValue, elements=node2Element, title='$\epsilon_{yy}$')

