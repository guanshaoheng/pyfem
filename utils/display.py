import matplotlib.pyplot as plt
import numpy as np

'''
        Author: Wang Jingzhou
        Email: Andrewwang@whu.edu.cn
        Affiliation: School of water resources and hydropower engineering, Wuhan University
'''

# display the mesh
def displayTriMesh(Trimesh):
    plt.figure()
    points_x = [Trimesh.point[point_name].coord[0]
                for point_name in Trimesh.point_name]
    points_y = [Trimesh.point[point_name].coord[1]
                for point_name in Trimesh.point_name]
    lines = []
    for triangle_name in Trimesh.triangle_name:
        triangle = Trimesh.triangle[triangle_name]
        for i in range(3):
            line = sorted([triangle.point_name[i], triangle.point_name[(i + 1) % 3]])
            if not(line in lines):
                lines.append(line)

        '''circle_center = triangle.circle_center
        plt.scatter(circle_center[0, 0],
                    circle_center[0, 1])
        t = np.linspace(0, 2 * np.pi, 100)
        x = circle_center[0, 0] + triangle.R * np.cos(t)
        y = circle_center[0, 1] + triangle.R * np.sin(t)
        plt.plot(x, y)'''

    for line in list(lines):
        line_x = [Trimesh.point[line[0]].coord[0],
                  Trimesh.point[line[1]].coord[0]]
        line_y = [Trimesh.point[line[0]].coord[1],
                  Trimesh.point[line[1]].coord[1]]
        plt.plot(line_x, line_y, color='k')
    plt.scatter(points_x, points_y, color='k')

    plt.axis('scaled')
    '''plt.xlim(-50, 150)
    plt.ylim(-50, 350)'''
    plt.show()

def displayQuadMesh(node, elem):
    plt.figure()
    points_x = node[:, :, 0]
    points_y = node[:, :, 1]
    lines = []
    for elem_i in elem:
        for j in range(4):
            a = elem_i[j]
            b = elem_i[(j + 1) % 4]
            line_x = [node[a[0], a[1], 0], node[b[0], b[1], 0]]
            line_y = [node[a[0], a[1], 1], node[b[0], b[1], 1]]
            plt.plot(line_x, line_y, color='k')
    plt.scatter(points_x, points_y, color='k')

    plt.axis('scaled')
    plt.show()