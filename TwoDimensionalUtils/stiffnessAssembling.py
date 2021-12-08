import matplotlib.pyplot as plt
import numpy as np
from TwoDimensionalUtils.ShapeFunction import TwoDimensionShape, plotElement, stiffnessLocal2Global


if __name__ == '__main__':
    ndim = 2
    Nnum = 4

    # -------------------------------------------------------------------
    # configuration 1 (node 11 element 6)
    nodeCoord = np.array([[0, 0], [1, 0], [1, 1],
                          [0, 1], [2, 0], [2, 1],
                          [2, 2], [1, 2], [0, 2],
                          [2, 3], [1, 3], [0, 3]], dtype=np.float)
    node2Element = np.array([[0, 1, 2, 3], [1, 4, 5, 2], [2, 5,6,7], [3,2,7,8],
                             [7, 6, 9, 10], [8, 7, 10, 11]])

    # configuration 2 (node 9 element 4)
    # nodeCoord = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1], [2,2], [1, 2], [0, 2]], dtype=np.float)
    # node2Element = np.array([[0, 1, 2, 3], [1, 4, 5, 2], [2, 5,6,7], [3,2,7,8]])
    # nodeCoord = np.array([[0, 0], [1, 0], [2, 0], [0, 1],
    #                       [1, 1], [2, 1], [0, 2], [1, 2],
    #                       [2, 2]], dtype=np.float)
    # node2Element = np.array([[1, 4, 3, 0], [2, 5, 4, 1], [5, 8, 7, 4], [7, 6, 3, 4]])

    # configuration 3 (node 8 element 3)
    # nodeCoord = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1], [3, 0], [3, 1]], dtype=np.float)
    # node2Element = np.array([[0, 1, 2, 3], [1, 4, 5, 2], [4,6,7,5]])

    # configuration 4 (node 6 element 2)
    # nodeCoord = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]], dtype=np.float)
    # node2Element = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])

    nodeNum, elementNum = len(nodeCoord), len(node2Element)
    nodeIndex = np.arange(nodeNum)
    elementIndex = np.arange(elementNum)

    shape = TwoDimensionShape()
    k = []
    for i in range(elementNum):
        k.append(shape.getElementStiffness(x=nodeCoord[node2Element[i]]))
    K_global = stiffnessLocal2Global(nodeIndex=nodeIndex, kList=k, node2Element=node2Element)

    # f = np.array([[1e6, 1e6], [-1e6, 1e6], [-1e6, -1e6], [1e6, -1e6]])
    f = np.zeros_like(nodeCoord, dtype=np.float)
    # f[2] = f[5] = f[8] = np.array([0, -1e6])
    f[9] =f[10]=f[11]= np.array([0, -2e6])
    # f[6] = np.array([-1e6, 0])
    # f[7] = np.array([-1e6, 0])
    # f[8] = np.array([-1e6, 0])

    mask = np.zeros_like(nodeCoord)
    mask[0] = np.array([1, 1])
    mask[4] = np.array([0, 1])
    # mask[3] = np.array([1, 1])
    # mask[4] = np.array([1, 1])
    # mask[7] = np.array([1, 1])
    # mask[4] = np.array([0, 1])
    uValue = mask * 0.
    u, f_node = shape.displacementBoundaryCondition(K_global=K_global, mask=mask, uValue=uValue, f=f)

    x_ = nodeCoord + u
    for i in range(elementNum):
        plotElement(nodeCoord[node2Element[i]], 'ro-')
        plotElement(x_[node2Element[i]], 'bo-')
    for j in range(nodeNum):
        plt.text(x=nodeCoord[j, 0], y=nodeCoord[j, 1], s=str(j))
    plt.axis('equal')
    plt.show()
