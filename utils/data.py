import numpy as np


class Nodes:
    def __init__(self, coord):
        """
            coord = [n_node, 2]
        :param coord:
        """
        self.n = len(coord)
        self.coord = coord


class Elements:
    def __init__(self, nodes2elements, name='rectangular'):
        """
            nodes2elements = [n_element, 4]
        """
        self.name = name
        self.nodes2elements = nodes2elements
        self.n = len(self.nodes2elements)