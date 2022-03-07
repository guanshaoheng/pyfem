import numpy as np
import math

'''
        Author: Wang Jingzhou
        Email: Andrewwang@whu.edu.cn
        Affiliation: School of water resources and hydropower engineering, Wuhan University
'''


# 点类
class Point(object):
    def __init__(self, name, coord):
        self.name = name
        self.coord = np.array(coord)

    def distance(self, point):
        vector = self.coord - point.coord
        distance = np.sqrt(np.dot(vector, vector.T))
        return distance


# 边类
class Line(object):
    def __init__(self, name, points):
        self.name = name
        self.point = points
        self.point_name = [point.name for point in points]


# 求解域类
class Domain(object):
    def __init__(self, lines):
        # 边界结构组
        self.line_name = set()
        self.line = dict()
        # 点结构组
        self.point_name = set()
        self.point = dict()
        for line in lines:
            self.line_name.add(line.name)
            self.line[line.name] = line
            for point in line.point:
                self.point_name.add(point.name)
                self.point[point.name] = point

    # 判断给定点是否在求解域内
    def insideJudge(self, point):
        O = point.coord
        D = np.zeros_like(O)
        D[0] = 1
        t_set = set()
        for line_name in self.line_name:
            points = self.line[line_name].point
            V = np.array([point_i.coord for point_i in points])
            for j in range(1, len(O)):
                V[j, :] -= V[0, :]
            T = O - V[0, :]
            V[0, :] = -D
            if round(np.linalg.det(V), 10) != 0:
                u = np.linalg.solve(np.mat(V).T, np.mat(T).T)
                if (u >= 0).all() and np.sum(u) - u[0] <= 1:
                    t_set.add(round(float(u[0]), 10))
        return len(t_set) % 2

    # 求解域边界细分
    def divideBoundary(self, type, value):
        origin_line_name = [line_name for line_name in self.line_name]
        for line_name in origin_line_name:
            point_1 = self.point[self.line[line_name].point_name[0]]
            point_2 = self.point[self.line[line_name].point_name[1]]
            if type == "number":
                n = value
            else:
                d = point_1.distance(point_2)
                n = math.ceil(d / value)
            new_points_list = [point_1.name]
            for i in range(n - 1):
                new_point_name = max(self.point_name) + 1
                new_line_name = max(self.line_name) + 1
                ratio = (i + 1) / n
                new_coord = (1 - ratio) * point_1.coord + ratio * point_2.coord
                new_point = Point(new_point_name, new_coord)
                new_line = Line(new_line_name, [self.point[new_points_list[-1]], new_point])
                self.point[new_point_name] = new_point
                self.point_name.add(new_point_name)
                self.line[new_line_name] = new_line
                self.line_name.add(new_line_name)
                new_points_list.append(new_point_name)
            new_line_name = max(self.line_name) + 1
            new_line = Line(new_line_name, [self.point[new_points_list[-1]], point_2])
            self.line[new_line_name] = new_line
            self.line_name.add(new_line_name)
            del self.line[line_name]
            self.line_name.discard(line_name)


# 三角形单元类
class Triangle(object):
    def __init__(self, name, points):
        self.name = name
        self.triangle = np.array([point.coord for point in points])
        D_1 = np.mat(self.triangle[1:, :] - self.triangle[:-1, :])
        # 向量叉乘为0，三点共线
        if np.linalg.det(D_1) == 0:
            print("该三角形三点共线，请校核。")
        else:
            # 逆时针排序检测
            if np.linalg.det(D_1) < 0:
                points[-2], points[-1] = points[-1], points[-2]
                self.triangle = np.array([point.coord for point in points])
            self.point_name = [point.name for point in points]
            self.point = {point.name:point for point in points}
            self.outsideCircle()
            self.center = np.sum(self.triangle, axis=0) / len(points)

    def outsideCircle(self):
        D_1 = np.mat(self.triangle[1:, :] - self.triangle[:-1, :])
        D_2 = np.mat(self.triangle[1:, :] + self.triangle[:-1, :])
        B = 1 / 2 * np.mat(np.diag(D_1 * D_2.T)).T
        self.circle_center = (np.linalg.inv(D_1) * B).T
        D_3 = self.circle_center - np.mat(self.triangle[0, :])
        self.R = float(np.sqrt(D_3 * D_3.T))

    # 判断点是否在三角形外接圆内
    def insideJudge(self, point):
        D = self.circle_center - np.mat(point.coord)
        r = float(np.sqrt(D * D.T))
        return r <= self.R

    # 判断点是否在三角形内
    def ifPointInside(self, point):
        O = point.coord
        V = np.array([self.point[point_name].coord
                      for point_name in self.point_name])
        for j in range(1, len(O) + 1):
            V[j, :] -= V[0, :]
        T = O - V[0, :]
        u = np.linalg.solve(np.mat(V[1:, :]).T, np.mat(T).T)
        if (u >= 0).all() and np.sum(u) <= 1:
            return 1
        else:
            return 0


# 三角形网格类
class Mesh(object):
    # 三角形网格的初始化
    # 需提供包含所有边界点的初始超级三角形
    def __init__(self, triangles):
        self.triangle = {triangle.name:triangle
                         for triangle in triangles}
        self.triangle_name = {triangle.name
                              for triangle in triangles}
        self.point_name = {point_name
                           for triangle in triangles
                           for point_name in triangle.point_name}
        self.point = {point_name: triangle.point[point_name]
                      for triangle in triangles
                      for point_name in triangle.point_name}

    def addPoint(self, point):
        # 找出需要删除的三角形
        triangle_del = []
        for name in self.triangle_name:
            triangle = self.triangle[name]
            if triangle.insideJudge(point):
                triangle_del.append(name)

        # 建立被删除三角形的顶点字典，保留顶点信息
        points_saved = dict()
        for name in triangle_del:
            points_saved.update(self.triangle[name].point)
        # 保留被删除三角形的边的信息
        # 被两个三角形单元共用的边需要删除无需保留
        lines = []
        for name in triangle_del:
            for i in range(3):
                point_1 = self.triangle[name].point_name[i]
                point_2 = self.triangle[name].point_name[(i + 1) % 3]
                line = sorted([point_1, point_2])
                # 加入当前选取的边，若该边已被记录过，
                # 说明该边为两个三角形共用，应删除
                if line in lines:
                    lines.remove(line)
                else:
                    lines.append(line)
        # 生成新的三角形网格
        new_triangle_list = []
        for line in lines:
            points = [points_saved[name] for name in line]
            points.append(point)
            points_test = np.array([point_i.coord
                                    for point_i in points])
            D_1 = np.mat(points_test[1:, :] - points_test[:-1, :])
            # 需要回避三单共线的情况
            if np.linalg.det(D_1) != 0:
                new_name = max(self.triangle_name) + 1
                new_triangle = Triangle(new_name, points)
                self.triangle[new_name] = new_triangle
                self.triangle_name.add(new_name)
                new_triangle_list.append(new_name)
        self.point_name.add(point.name)
        self.point[point.name] = point
        # 删除旧的需要删除的三角形网格
        for name in triangle_del:
            del self.triangle[name]
            self.triangle_name.discard(name)
        return new_triangle_list, triangle_del

    # 删除点，用于移除超级三角形
    def delPoint(self, point):
        point_name = point.name
        triangle_del = []
        # 确定需要移除的三角形
        for triangle_name in self.triangle_name:
            tetra = self.triangle[triangle_name]
            if point_name in tetra.point_name:
                triangle_del.append(triangle_name)
        # 移除相应的三角形
        for triangle_name in triangle_del:
            self.delTriangle(self.triangle[triangle_name])
        # 删除相应的点
        self.point_name.discard(point_name)
        del self.point[point_name]

        return point

    # 删除多余的三角形
    def delTriangle(self, triangle):
        if triangle.name in self.triangle_name:
            del self.triangle[triangle.name]
            self.triangle_name.discard(triangle.name)

    # 初始化长度标尺和单元无量纲半径
    def initMeshParam(self, domain):
        self.point_value = dict()
        point_to_point = dict()

        # 寻找边界上相邻点
        for line_name in domain.line_name:
            line = domain.line[line_name]
            for point_name in line.point_name:
                if not(point_name in point_to_point):
                    point_to_point[point_name] = set()
                for link_point in line.point_name:
                    point_to_point[point_name].add(link_point)

        # 计算边界点长度标尺
        for point_name in domain.point_name:
            self.point_value[point_name] = 0
            point_to_point[point_name].discard(point_name)
            for link_point in list(point_to_point[point_name]):
                self.point_value[point_name]\
                    += domain.point[point_name].\
                    distance(domain.point[link_point])
            self.point_value[point_name] *= np.sqrt(3) / 2 \
                                            / len(point_to_point[point_name])

        # 计算单元外接圆圆心长度标尺及无量纲半径
        self.circle_value = dict()
        self.R = dict()
        for triangle_name in self.triangle_name:
            self.circle_value[triangle_name], self.R[triangle_name] \
                = self.calCircleValue(triangle_name)

    # 计算给定单元的圆心长度标尺和单元的无量纲半径
    def calCircleValue(self, triangle_name):
        triangle = self.triangle[triangle_name]
        circle_center = Point(max(self.point_name) + 1,
                              np.array(triangle.circle_center).flatten())
        L = 10 ** 100
        if triangle.ifPointInside(circle_center):
            L = self.calPointValue(triangle_name, circle_center)
        else:
            for triangle_i in self.triangle_name:
                if self.triangle[triangle_i].ifPointInside(circle_center):
                    L = self.calPointValue(triangle_i, circle_center)
                    break
        R = triangle.R / L
        return L, R

    # 计算点的长度标尺
    def calPointValue(self, triangle_name, point):
        triangle = self.triangle[triangle_name]
        up = 0
        down = 0
        for point_name in triangle.point_name:
            l = point.distance(self.point[point_name])
            up += self.point_value[point_name] / l
            down += 1 / l
        L = up / down
        return L

    # 更新长度标尺和单元无量纲半径
    def updateMeshParam(self, triangle_name):
        triangle = self.triangle[triangle_name]
        circle_center = np.array(triangle.circle_center).flatten()
        new_point = Point(max(self.point_name) + 1, circle_center)
        # 计算新点的长度标尺
        self.point_value[new_point.name] = self.circle_value[triangle_name]
        # 加入新点
        new_triangle_list, triangle_del = self.addPoint(new_point)
        for new_triangle_name in new_triangle_list:
            self.circle_value[new_triangle_name], self.R[new_triangle_name] \
                = self.calCircleValue(new_triangle_name)
        for triangle_name_del in triangle_del:
            del self.circle_value[triangle_name_del]
            del self.R[triangle_name_del]


