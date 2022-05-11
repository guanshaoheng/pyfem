import matplotlib.pyplot as plt

def displayQuadMesh(node, elem, f=list(), ac=2):
    plt.figure()
    points_x = node[:, 0]
    points_y = node[:, 1]
    for elem_i in elem:
        for j in range(4):
            a = elem_i[j]
            b = elem_i[(j + 1) % 4]
            line_x = [node[a, 0], node[b, 0]]
            line_y = [node[a, 1], node[b, 1]]
            plt.plot(line_x, line_y, color='k')
    if len(f):
        for i, node_i in enumerate(node):
            if f[i] > 1e-5:
                plt.text(x=node_i[0], y=node_i[1],
                         s=str(round(f[i], ac)),
                         color="r")
            elif f[i] < -1e-5:
                plt.text(x=node_i[0], y=node_i[1],
                         s=str(round(f[i], ac)),
                         color="g")
            else:
                plt.text(x=node_i[0], y=node_i[1],
                         s=str(round(f[i], ac)),
                         color="b")
    plt.scatter(points_x, points_y, color='k')
    plt.axis('scaled')
    plt.show()