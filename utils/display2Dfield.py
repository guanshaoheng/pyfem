import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


# converts quad elements into tri elements
def quads_to_tris(quads):
    tris = [[None for j in range(3)] for i in range(2*len(quads))]
    for i in range(len(quads)):
        j = 2*i
        n0 = quads[i][0]
        n1 = quads[i][1]
        n2 = quads[i][2]
        n3 = quads[i][3]
        tris[j][0] = n0
        tris[j][1] = n1
        tris[j][2] = n2
        tris[j + 1][0] = n2
        tris[j + 1][1] = n3
        tris[j + 1][2] = n0
    return tris


# plots a finite element mesh
def plot_fem_mesh(nodes_x, nodes_y, elements):
    for element in elements:
        x = [nodes_x[element[i]] for i in range(len(element))]
        y = [nodes_y[element[i]] for i in range(len(element))]
        plt.fill(x, y, edgecolor='black', fill=False)


def plot_node_displacement(nodes_x, nodes_y, nodal_values, elements, title=None):
    plot_fem_mesh(nodes_x, nodes_y, elements)
    if len(elements[0]) == 4:
        elements_all_tris = quads_to_tris(elements)
    else:
        elements_all_tris = elements
    triangulation = tri.Triangulation(nodes_x, nodes_y, elements_all_tris)
    # plot the finite element mesh
    # plot the contours
    plt.tricontourf(triangulation, nodal_values)
    if title is not None:
        plt.title(title, fontsize=15)
    # show
    plt.colorbar()
    xmin, xmax = np.min(nodes_x), np.max(nodes_x)
    ymin, ymax = np.min(nodes_y), np.max(nodes_y)
    lx, ly = xmax-xmin, ymax-ymin
    plt.xlim([xmin-0.1*lx, xmax+0.1*lx])
    plt.ylim([ymin-0.1*ly, ymax+0.1*ly])
    # plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    # FEM data
    nodes_x = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]
    nodes_y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    nodal_values = [1.0, 0.9, 1.1, 0.9, 2.1, 2.1, 0.9, 1.0, 1.0, 0.9, 0.8]
    elements_tris = [[2, 6, 5], [5, 6, 10], [10, 9, 5]]
    elements_quads = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 8, 7], [4, 5, 9, 8]]
    # convert all elements into triangles
    elements_all_tris = elements_tris + quads_to_tris(elements_quads)
    plot_node_displacement(nodes_x, nodes_y, nodal_values, elements_all_tris)




