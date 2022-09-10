import matplotlib.pyplot as plt
import numpy as np
from data import Nodes, Elements


class Domain:
    def __init__(self, lx=1.0, ly=2.0, nx=11, ny=21):
        self.lx, self.ly, self.nx, self.ny = lx, ly, nx, ny
        self.nodes, self.elements = self.mesh()
        self.n_nodes = self.nodes.n
        self.n_elements = self.elements.n
        self.plot_elements()
        self.u = np.zeros(shape=[self.n_nodes, 2])

    def mesh(self):
        dx, dy = self.lx/(self.nx-1), self.ly/(self.ny-1)
        node_coord = np.zeros(shape=(self.nx*self.ny, 2))
        for i in range(self.nx):
            for j in range(self.ny):
                node_coord[i + j*self.nx] = np.array([dx*i, dy*j])
        node2element = np.zeros(shape=((self.nx-1) * (self.ny-1), 4), dtype=int)
        for i in range(self.nx-1):
            for j in range(self.ny-1):
                node2element[i+(self.nx-1)*j] = np.array([i+(j*(self.nx)), i+(j*(self.nx))+1, i+(1+j)*(self.nx)+1, i+((1+j)*(self.nx))])

        nodes = Nodes(node_coord)
        elements = Elements(node2element)
        return nodes, elements

    def plot_elements(self):
        for index in self.elements.nodes2elements:
            plt.plot(self.nodes.coord[index, 0], self.nodes.coord[index, 1], 'b')
        plt.plot([0, 0], [0, np.max(self.nodes.coord[:, 1])], 'b')
        plt.axis("equal")
        plt.show()

    def get_A(self, ):



if __name__== "__main__":
    domain_o = Domain()