from TwoDimensionalUtils.data import *
import numpy as np
# from display import displayMesh

'''
        Author: Wang Jingzhou
        Email: Andrewwang@whu.edu.cn
        Affiliation: School of water resources and hydropower engineering, Wuhan University
'''

def mesh(domain):
    points = np.array([domain.point[point_name].coord
                       for point_name in domain.point_name])
    # get a rectangle cover the points
    coord_max = np.max(points, axis=0)
    coord_min = np.min(points, axis=0)

    # enlarge the rectangle
    ratio = 3
    length = coord_max - coord_min
    coord_max = coord_max + ratio * length
    coord_min = coord_min - ratio * length
    length = ( 1 + 2 * ratio) * length

    # create a super triangle according to the rectangle

    new_name = max(domain.point_name) + 1
    points = [Point(new_name, coord_min)]
    for i in range(len(coord_max)):
        new_name += 1
        offset = np.zeros_like(coord_min)
        offset[i] = len(coord_min) * length[i]
        points.append(Point(new_name, coord_min + offset))
    init_triangles = [Triangle(0, points)]

    # initialize the mesh
    mesh = Mesh(init_triangles)

    # add the points at the boundary of the domain
    for point_name in domain.point_name:
        mesh.addPoint(domain.point[point_name])

    # delete the points which in the super triangle
    point_init = {init_triangle.point[point_name]
                  for init_triangle in init_triangles
                  for point_name in init_triangle.point_name}
    for point in list(point_init):
        mesh.delPoint(point)

    # delete the element out of the domain
    triangle_del = []
    for triangle_name in mesh.triangle_name:
        triangle = mesh.triangle[triangle_name]
        center_coord = triangle.center
        point = Point(0, center_coord)
        if not(domain.insideJudge(point)):
            triangle_del.append(triangle_name)
    for triangle_name in triangle_del:
        mesh.delTriangle(mesh.triangle[triangle_name])

    # optimize the mesh
    mesh.initMeshParam(domain)
    mesh_R = sorted(zip(mesh.R.values(),mesh.R.keys()))
    while mesh_R[-1][0] > 1:
        mesh.updateMeshParam(mesh_R[-1][1])
        mesh_R = sorted(zip(mesh.R.values(), mesh.R.keys()))

    return mesh

