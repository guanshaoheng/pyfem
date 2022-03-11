import matplotlib.pyplot as plt
# import numpy as np

'''
        Author: Wang Jingzhou
        Email: Andrewwang@whu.edu.cn
        Affiliation: School of water resources and hydropower engineering, Wuhan University
'''

# display the mesh
def displayMesh(mesh):
    plt.figure()
    points_x = [mesh.point[point_name].coord[0]
                for point_name in mesh.point_name]
    points_y = [mesh.point[point_name].coord[1]
                for point_name in mesh.point_name]
    lines = []
    for triangle_name in mesh.triangle_name:
        triangle = mesh.triangle[triangle_name]
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
        line_x = [mesh.point[line[0]].coord[0],
                  mesh.point[line[1]].coord[0]]
        line_y = [mesh.point[line[0]].coord[1],
                  mesh.point[line[1]].coord[1]]
        plt.plot(line_x, line_y, color='k')
    plt.scatter(points_x, points_y, color='k')

    plt.axis('scaled')
    '''plt.xlim(-50, 150)
    plt.ylim(-50, 350)'''
    plt.show()