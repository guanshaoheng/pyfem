import numpy as np
import sympy as sym

class ElementType(object):
    def __init__(self, node_list, x, cons):
        self.node_list = node_list
        self.x = x
        self.n_dim = cons.n_dim
        self.n_node, self.n_dim = np.shape(x)
        self.getGaussian()
        self.shapeFunction()
        self.getN_diff()
        self.cons = cons
        self.getElementStiffness()

    def getGaussian(self):
        self.n_gauss = self.n_node
        self.gaussian_points = np.zeros(
            shape=(self.n_gauss, self.n_dim))
        self.W = np.zeros(self.n_gauss)

    def shapeFunction(self):
        self.basis = sym.Matrix([])
        self.N = sym.Matrix([])

    def getN_diff(self):
        self.N_diff = self.N.jacobian(self.basis)
        self.N_array = np.zeros(shape=(self.n_gauss, self.n_node))
        self.N_d_local = np.zeros(shape=(self.n_gauss, self.n_node, self.n_dim))
        for i, gaussian_point in enumerate(self.gaussian_points):
            subs_basis = [(self.basis[i], gaussian_point[i])
                          for i in range(self.n_dim)]
            self.N_array[i] = np.array(self.N.subs(subs_basis)).astype(np.float32).reshape(-1)
            self.N_d_local[i] = self.N_diff.subs(subs_basis)

    def getElementStiffness(self):
        je = np.einsum('pni,nj->pji', self.N_d_local, self.x)
        je_det = np.linalg.det(je)
        je_inv = np.linalg.inv(je)
        self.N_d_global = np.einsum('pmj,pji->pmi', self.N_d_local, je_inv)
        self.K_element = np.einsum('pmi, pijkl, pnk, p, p->mjnl',
                                   self.N_d_global, self.cons.Dep,
                                   self.N_d_global, je_det, self.W)

    # 积分点处的应力和应变
    def getDEAndDS(self, d_u):
        d_e = (np.einsum("pki, kj->pij", self.N_d_global, d_u) +
             np.einsum("pki, kj->pji", self.N_d_global, d_u)) / 2
        # TODO 进入本构的计算
        d_s = np.einsum("pijkl, pkl->pij", self.cons.Dep, d_e)
        return d_e, d_s

    def plasticJudge(self, d_u):
        d_e, d_s = self.getDEAndDS(d_u)
        r = self.cons.calculateR(d_s)
        return r

    def updateElementStiffness(self, d_u):
        self.x += d_u
        je = np.einsum('pni,nj->pji', self.N_d_local, self.x)
        je_det = np.linalg.det(je)
        je_inv = np.linalg.inv(je)
        self.N_d_global = np.einsum('pmj,pji->pmi', self.N_d_local, je_inv)
        self.K_element = np.einsum('pmi, pijkl, pnk, p, p->mjnl',
                                   self.N_d_global, self.cons.Dep,
                                   self.N_d_global, je_det, self.W)

class Q4(ElementType):
    def getGaussian(self):
        self.n_gauss = 4
        temp = np.sqrt(1 / 3)
        self.gaussian_points = np.array([[-temp, -temp], [temp, -temp],
                                        [temp, temp], [-temp, temp]])
        self.W = np.array([1, 1, 1, 1])

    def shapeFunction(self):
        xi, eta = sym.symbols('xi, eta')
        self.basis = sym.Matrix([xi, eta])
        N1 = (1 - xi) * (1 - eta) / 4
        N2 = (1 + xi) * (1 - eta) / 4
        N3 = (1 + xi) * (1 + eta) / 4
        N4 = (1 - xi) * (1 + eta) / 4
        self.N = sym.Matrix([N1, N2, N3, N4])

    def getV(self):
        v = 0
        for i in range(self.n_node - 1):
            v += np.linalg.det(self.x[i:(i + 2)])
        v += np.linalg.det(np.array([self.x[-1], self.x[0]]))
        v /= 2
        return abs(v)

class CST(ElementType):
    def getGaussian(self):
        self.n_gauss = 3
        self.gaussian_points = np.array([[0.5, 0], [0, 0.5], [0.5, 0.5]])
        self.W = np.array([1 / 6, 1 / 6, 1 / 6])

    def shapeFunction(self):
        xi, eta = sym.symbols('xi, eta')
        self.basis = sym.Matrix([xi, eta])
        N1 = 1 - xi - eta
        N2 = xi
        N3 = eta
        self.N = sym.Matrix([N1, N2, N3])

class C3D8(ElementType):
    def getGaussian(self):
        self.n_gauss = 8
        temp = np.sqrt(1 / 3)
        self.gaussian_points = np.array([[-temp, -temp, -temp], [temp, -temp, -temp],
                                         [temp, temp, -temp], [-temp, temp, -temp],
                                         [-temp, -temp, temp], [temp, -temp, temp],
                                         [temp, temp, temp], [-temp, temp, temp]])
        self.W = np.array([1, 1, 1, 1,
                           1, 1, 1, 1])

    def shapeFunction(self):
        xi, eta, zeta = sym.symbols('xi, eta, zeta')
        self.basis = sym.Matrix([xi, eta, zeta])
        N1 = (1 - xi) * (1 - eta) * (1 - zeta) / 8
        N2 = (1 + xi) * (1 - eta) * (1 - zeta) / 8
        N3 = (1 + xi) * (1 + eta) * (1 - zeta) / 8
        N4 = (1 - xi) * (1 + eta) * (1 - zeta) / 8
        N5 = (1 - xi) * (1 - eta) * (1 + zeta) / 8
        N6 = (1 + xi) * (1 - eta) * (1 + zeta) / 8
        N7 = (1 + xi) * (1 + eta) * (1 + zeta) / 8
        N8 = (1 - xi) * (1 + eta) * (1 + zeta) / 8
        self.N = sym.Matrix([N1, N2, N3, N4,
                             N5, N6, N7, N8])

class TET4(ElementType):
    def getGaussian(self):
        self.n_gauss = 4
        self.gaussian_points = np.array([[0.5, 0, 0], [0, 0.5, 0],
                                         [0, 0, 0.5], [0.5, 0.5, 0.5]])
        self.W = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6])

    def shapeFunction(self):
        xi, eta, zeta = sym.symbols('xi, eta, zeta')
        self.basis = sym.Matrix([xi, eta, zeta])
        N1 = 1 - xi - eta - zeta
        N2 = xi
        N3 = eta
        N4 = zeta
        self.N = sym.Matrix([N1, N2, N3, N4])