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
    # 获取点云外包矩形
    coord_max = np.max(points, axis=0)
    coord_min = np.min(points, axis=0)

    # 外包盒扩大
    # 要设大点才能确保边界匹配上，ratio如果太小，边界匹配会出问题
    ratio = 3
    length = coord_max - coord_min
    coord_max = coord_max + ratio * length
    coord_min = coord_min - ratio * length
    length = ( 1 + 2 * ratio) * length

    # 根据放大的外包矩形确定外包超级三角形顶点

    new_name = max(domain.point_name) + 1
    points = [Point(new_name, coord_min)]
    # 为了让coord_max有点用，所以用的是coord_max
    for i in range(len(coord_max)):
        new_name += 1
        offset = np.zeros_like(coord_min)
        offset[i] = len(coord_min) * length[i]
        points.append(Point(new_name, coord_min + offset))
    init_triangles = [Triangle(0, points)]

    # 网格初始化
    mesh = Mesh(init_triangles)

    # 将边界点依次加入到初始化后的网格中
    for point_name in domain.point_name:
        mesh.addPoint(domain.point[point_name])

    # 删去初始引入的超级三角形
    point_init = {init_triangle.point[point_name]
                  for init_triangle in init_triangles
                  for point_name in init_triangle.point_name}
    for point in list(point_init):
        mesh.delPoint(point)

    # 删去边界外的多余三角形
    triangle_del = []
    for triangle_name in mesh.triangle_name:
        triangle = mesh.triangle[triangle_name]
        center_coord = triangle.center
        point = Point(0, center_coord)
        if not(domain.insideJudge(point)):
            triangle_del.append(triangle_name)
    for triangle_name in triangle_del:
        mesh.delTriangle(mesh.triangle[triangle_name])

    # 三角单元细分
    # 初始化单元无量纲半径
    mesh.initMeshParam(domain)
    mesh_R = sorted(zip(mesh.R.values(),mesh.R.keys()))
    while mesh_R[-1][0] > 1:
        mesh.updateMeshParam(mesh_R[-1][1])
        mesh_R = sorted(zip(mesh.R.values(), mesh.R.keys()))

    return mesh

