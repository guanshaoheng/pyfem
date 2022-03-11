import numpy as np
import sympy as sym
import matplotlib.pyplot as plt


class TwoDimensionShape:
    def __init__(self):
        self.ndim = 2
        self.Nnum = 3
        self.gaussianPoints = self.getGaussian()
        self.N, self.N_diff_local = self.getN_diff()
        self.N_diff_global = self.N_diff_local
        self.elementStiffness = self.getElementStiffness()

    def getGaussian(self):
        temp = 0.5
        return np.array([[temp, 0], [temp, temp], [0, temp]])

    def getN_diff(self):
        xi, eta = sym.symbols('xi, eta')
        basis = sym.Matrix([xi, eta])
        N1 = 1 - xi - eta
        N2 = xi
        N3 = eta
        N = sym.Matrix([N1, N2, N3])
        N_diff = N.jacobian(basis)
        N_array = np.zeros(shape=(self.Nnum, self.Nnum))
        N_d_array = np.zeros(shape=(self.Nnum, self.Nnum, self.ndim))
        for i, gaussianPoint in enumerate(self.gaussianPoints):
            N_array[i] = np.array(N.subs([(xi, gaussianPoint[0]), (eta, gaussianPoint[1])])).astype(np.float32).reshape(-1)
            N_d_array[i] = N_diff.subs([(xi, gaussianPoint[0]), (eta, gaussianPoint[1])])
        return N_array, N_d_array

    def getElementStiffness(self, x=None, D=None):
        if x is None:
            x = np.array([[1, 0], [0, 1], [0, 0]])
        if D is None:
            E, mu = 1e8, 0.2
            lam, G = E * mu / (1 + mu) / (1 - 2 * mu), 0.5 * E / (1 + mu)
            D = np.zeros(shape=(2, 2, 2, 2))
            D[0, 0, 0, 0] = D[1, 1, 1, 1] = lam + G * 2
            D[0, 0, 1, 1] = D[1, 1, 0, 0] = lam
            D[0, 1, 0, 1] = D[0, 1, 1, 0] = D[1, 0, 0, 1] = D[1, 0, 1, 0] = G
        je = np.einsum('ni,pnj->pij', x, self.N_diff_local)
        je_det =  np.einsum('i,i->i', np.linalg.det(je), np.array([1/6, 1/6, 1/6]))
        je_inv = np.linalg.inv(je)
        self.N_diff_global = np.einsum('pmj,pji->pmi', self.N_diff_local, je_inv)
        K_element = np.einsum('pmi,ijkl,pnk,p->mjnl', self.N_diff_global, D, self.N_diff_global, je_det)
        return K_element

    def displacementBoundaryCondition(self, K_global, mask, f):
        mask_free = np.equal(mask, 0)
        K_free = K_global[mask_free][:, mask_free]
        f_free = f[mask_free]
        return K_free, f_free

    def getStrain(self, u, node2Element, nodeCoord):
        uElementNode = u[node2Element]
        coordElementNode = nodeCoord[node2Element]
        epsilon = np.einsum('pmi, qmj->pqij', uElementNode, self.N_diff_global)
        epsilon = 0.5*(epsilon + np.einsum('pqij->pqji', epsilon))
        epsilonCoord = np.einsum('pmi, qm->pqi', coordElementNode, self.N)
        return epsilon, epsilonCoord

    def solve(self, mask, K_global, K_free, f_free, uValue):
        mask_free = np.equal(mask, 0)
        nNode = mask_free.shape[0]
        u_free = np.linalg.solve(K_free, f_free)
        u = np.zeros_like(mask_free, dtype=np.float)
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

    def interpolateStrainToNode(self, nodecoord, node2Elment, epsilon):
        nNode = len(nodecoord)
        epsilonNodeAdd = np.zeros(shape=(nNode, self.ndim, self.ndim))
        nodeAddNum = np.zeros(shape=(nNode))
        N_inv = np.linalg.inv(self.N)
        for i, element in enumerate(node2Elment):
            epsilonElement = epsilon[i]
            epsilonNodeAdd[element] += np.einsum('mn, mij->nij', N_inv, epsilonElement)
            nodeAddNum[element] += 1
        nodeAddNum_inv = 1/nodeAddNum
        epsilonNode = np.einsum('pij, p->pij', epsilonNodeAdd, nodeAddNum_inv)
        return epsilonNode

def stiffnessAssembling(nodeIndex, kList, node2Element, ndim=2, Nnum=3):
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


def plotGuassian(coord, *args):
    for gaussianPoints in coord:
        plt.plot(gaussianPoints[:, 0], gaussianPoints[:, 1], *args)
    for i, coorGaussian in enumerate(coord[0]):
        plt.text(x=coorGaussian[0], y=coorGaussian[1], s=str(i))
    return

def twoDimFEM(mesh, f, mask):
    # 从网格获取计算单元节点信息
    nodeCoord = np.array([mesh.point[point_name].coord
                          for point_name in sorted(mesh.point_name)])
    node2Element = np.array([mesh.triangle[triangle_name].point_name
                             for triangle_name in mesh.triangle_name]) - 1
    # 获取单元节点数量
    nodeNum, elementNum = len(nodeCoord), len(node2Element)
    nodeIndex = np.arange(nodeNum)
    elementIndex = np.arange(elementNum)

    shape = TwoDimensionShape()
    k = []
    for i in range(elementNum):
        k.append(shape.getElementStiffness(x=nodeCoord[node2Element[i]]))
    K_global = stiffnessAssembling(nodeIndex=nodeIndex, kList=k, node2Element=node2Element)

    uValue = mask * 0.
    k_free, f_free = shape.displacementBoundaryCondition(K_global=K_global, mask=mask, f=f)
    u, f_node = shape.solve(mask=mask, K_global=K_global, K_free=k_free, f_free=f_free, uValue=uValue)
    epsilon, gaussianCoord = shape.getStrain(u=u, node2Element=node2Element, nodeCoord=nodeCoord + u)
    epsilonNode = shape.interpolateStrainToNode(nodeCoord, node2Element, epsilon)

    x_ = nodeCoord + u
    # plotGuassian(gaussianCoord, 'ro')
    for i in range(elementNum):
        plotElement(nodeCoord[node2Element[i]], 'ko-')
        plotElement(x_[node2Element[i]], 'bo-')
    for j in range(nodeNum):
        plt.text(x=nodeCoord[j, 0], y=nodeCoord[j, 1], s=str(j))
        # plt.text(x=x_[j, 0], y=x_[j, 1], s=str(j))
    plt.axis('equal')
    plt.show()