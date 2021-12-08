import numpy as np
import sympy
import sympy as sym
import matplotlib.pyplot as plt
import scipy as sp


class TwoDimensionShape:
    def __init__(self):
        self.ndim = 2
        self.Nnum = 4
        self.gaussianPoints = self.getGaussian()
        self.N, self.N_diff_local = self.getN_diff()
        self.elementStiffness = self.getElementStiffness()

    def getGaussian(self):
        temp = np.sqrt(1 / 3)
        return np.array([[-temp, -temp], [temp, -temp], [temp, temp], [-temp, temp]])

    def getN_diff(self):
        xi, eta = sym.symbols('xi, eta')
        basis = sym.Matrix([xi, eta])
        N1 = (1 - xi) * (1 - eta) / 4
        N2 = (1 + xi) * (1 - eta) / 4
        N3 = (1 + xi) * (1 + eta) / 4
        N4 = (1 - xi) * (1 + eta) / 4
        N = sym.Matrix([N1, N2, N3, N4])
        N_diff = N.jacobian(basis)
        N_array = np.zeros(shape=(self.Nnum, self.Nnum))
        N_d_array = np.zeros(shape=(self.Nnum, self.Nnum, self.ndim))
        for i, gaussianPoint in enumerate(self.gaussianPoints):
            N_array[i] = np.array(N.subs([(xi, gaussianPoint[0]), (eta, gaussianPoint[1])])).astype(np.float32).reshape(-1)
            N_d_array[i] = N_diff.subs([(xi, gaussianPoint[0]), (eta, gaussianPoint[1])])
        return N_array, N_d_array

    def getElementStiffness(self, x=None, D=None):
        if x is None:
            # x = np.array([[-0.8, 0.], [0, -0.8], [0.8, 0], [0, 0.8]])
            x = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        if D is None:
            E, mu = 1e8, 0.2
            lam, G = E * mu / (1 + mu) / (1 - 2 * mu), 0.5 * E / (1 + mu)
            D = np.zeros(shape=(2, 2, 2, 2))
            D[0, 0, 0, 0] = lam + G * 2
            D[1, 1, 1, 1] = lam + G * 2
            # D[0, 0, 1, 1] = D[1, 1, 0, 0] = lam
            # D[0, 1, 0, 1] = D[0, 1, 1, 0] = D[1, 0, 0, 1] = D[1, 0, 1, 0] = G
            D[0, 0, 1, 1] = D[1, 1, 0, 0] = lam
            D[0, 1, 0, 1] = D[0, 1, 1, 0] = D[1, 0, 0, 1] = D[1, 0, 1, 0] = G
        je = np.einsum('ni,pnj->pij', x, self.N_diff_local)  # pij
        je_det = np.linalg.det(je)  # p
        je_inv = np.linalg.inv(je)  # pij -> pji
        N_diff_global = np.einsum('pmj,pji->pmi', self.N_diff_local, je_inv)
        # NOTE: VERY IMPORTANT!!!!!
        K_element = np.einsum('pmk,ikjl,pnl,p->minj', N_diff_global, D, N_diff_global, je_det)
        # K_element = np.einsum('pmk,ijkl,pnl,p->minj', N_diff_global, D, N_diff_global, je_det)
        return K_element

    def displacementBoundaryCondition(self, K_global, mask, uValue, f):
        mask_free = np.equal(mask, 0)
        nNode = mask.shape[0]
        K_free = K_global[mask_free][:, mask_free]
        f_free = f[mask_free]
        u_free = np.linalg.solve(K_free, f_free)
        u = np.zeros_like(mask, dtype=np.float)
        tempPointer = 0
        for i in range(nNode):
            for j in range(self.ndim):
                if mask_free[i, j]:
                    u[i, j] = u_free[tempPointer]
                    tempPointer += 1
                else:
                    u[i, j] = uValue[i, j]
        f_calculated = np.einsum('minj, nj->mi', K_global, u)
        return u, f_calculated


def stiffnessLocal2Global(nodeIndex, kList, node2Element, ndim=2, Nnum=4):
    nodeNum = len(nodeIndex)
    elementNum = len(node2Element)
    k_global = np.zeros(shape=(nodeNum, ndim, nodeNum, ndim), dtype=np.float)
    for p in range(elementNum):
        ktemp = kList[p]
        elementNode = node2Element[p]
        for m in range(Nnum):
            for n in range(Nnum):
                k_global[elementNode[m], :, elementNode[n], :] += ktemp[m, :, n, :]
    return k_global


def plotElement(coord, *args):
    coord = np.concatenate((coord, coord[0:1]))
    plt.plot(coord[:, 0], coord[:, 1], *args)
    return


if __name__ == "__main__":
    # ----------------------------------------------------
    # shape function
    a = TwoDimensionShape()
    nodeCoord = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=np.float)
    node2Element = np.array([[0, 1, 2, 3]])
    elementNum = len(node2Element)
    nodeIndex = np.arange(len(nodeCoord))
    x = nodeCoord[node2Element[0]]
    k = []
    for i in range(elementNum):
        k.append(a.getElementStiffness(x=nodeCoord[node2Element[i]]))
    K_global = stiffnessLocal2Global(nodeIndex=nodeIndex, kList=k, node2Element=node2Element)

    ndim = x.shape[-1]
    nNode = x.shape[0]

    # f = np.array([[0, 0], [0, 0], [-1e6, -0], [0, 0]])
    f = np.zeros_like(x, dtype=np.float)
    f[2] = np.array([-1e6, +0])

    # ----------------------------------------------------
    # boundary conditions
    mask_constrained = np.zeros_like(x)
    mask_constrained[0] = np.array([1, 1])  # where the 1st boundary condition is applied
    mask_constrained[1] = np.array([0, 1])  # where the 1st boundary condition is applied
    uValue = mask_constrained*0.
    u, f_node = a.displacementBoundaryCondition(K_global, mask_constrained, uValue, f)

    x_ = nodeCoord+u
    plotElement(nodeCoord, 'ro-')
    plotElement(x_, 'bo-')
    for i in range(4):
        plt.text(x=nodeCoord[i, 0], y=nodeCoord[i, 1], s='%d' % i, fontsize=15)
    plt.axis('equal')
    plt.show()

