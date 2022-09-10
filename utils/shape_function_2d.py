import numpy as np


class shape_function_2d:
    def __init__(self, ):
        self.gauss_coord = self.get_gauss_coord()
        self.N = np.array(
            [self.get_N(xi=i[0], eta=i[1]) for i in self.gauss_coord])
        self.dN_dx = np.array(
            [self.get_dN_dx(xi=i[0], eta=i[1]) for i in self.gauss_coord])
        print()

    def get_gauss_coord(self):
        temp = np.sqrt(1./3.)
        gauss_coord = np.array([
            [-temp, -temp],
            [temp, -temp],
            [temp, temp],
            [-temp, temp]
        ])
        return gauss_coord

    def get_N(self, xi, eta):
        N = 0.25 * np.array(
            [(1 - xi) * (1 - eta),
             (1 + xi) * (1 - eta),
             (1 + xi) * (1 + eta),
             (1 - xi) * (1 + eta)])
        return N

    def get_dN_dx(self, xi, eta):
        dN_dx = 0.25 * np.array([
            [-(1 - eta), -(1 - xi)],
            [(1 - eta), -(1 + xi)],
            [(1 + eta), (1 + xi)],
            [-(1 + eta), (1 - xi)]
        ])
        return dN_dx

    def get_stiffness_element(self, x_global):
        je = np.einsum("pmi, mj->pij", self.dN_dx, x_global)
        je_det = np.linalg.det(je)
        je_inv = np.linalg.inv(je)
        dN_dX = np.einsum("pmj, pij->pmi", self.dN_dx, je_inv)  # mi
        A = np.einsum("pmi, pni, p -> mn", dN_dX, dN_dX, je_det)*je_det
        return A

if __name__ == "__main__":
    a = shape_function_2d()
    aa = a.get_stiffness_element(
        x_global=np.array([
        [0, 0,], [1, 0], [1, 1], [0, 1]
        ]))
