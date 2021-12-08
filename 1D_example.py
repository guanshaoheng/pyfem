import numpy as np
import scipy.integrate.quadrature as integrator

"""
        An 1-dimensional linear problem is used to describe the FEM process
        
        reference:
            [1] https://www.youtube.com/watch?v=rdaZuKFK-4k
        
"""


class OneDimensionalProblem:
    def __init__(self):
        self.NodeNum = 5
        self.elementNum = 4
        self.nodeCoordinate = np.linspace(0, 1, 5)
        self.element = [[i, i+1] for i in range(self.NodeNum-1)]
        self.gaussionNorm = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        self.gaussionGlobal = [self.mapGaussian2Local(self.nodeCoordinate[i[0]], self.nodeCoordinate[i[1]]) for i in self.element]
        print()

    def shapeFunction(self, x, x1, x2):
        w1 = (x2-x)/(x2-x1)
        w2 = (x-x1)/(x2-x1)
        return np.array([w1, w2])

    def shapeFunctionDx(self, x1, x2):
        dx1 = -1/(x2-x1)
        dx2 = 1/(x2-x1)
        return np.array([dx1, dx2])

    def mapGaussian2Local(self, x1, x2):
        gaussionLocal = np.zeros_like(self.gaussionNorm)
        for i in range(len(self.gaussionNorm)):
            gaussionLocal[i] = (self.gaussionNorm[i]+1)/2*(x2-x1)+x1
        return gaussionLocal


if __name__ == '__main__':
    oneDimProblem = OneDimensionalProblem()