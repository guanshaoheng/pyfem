import numpy as np
import sympy as sym

# Define a class of the solver
# You are supposed to give the number of the dimensions
# ,the number of the nodes to create the solver
class Solver(object):
    def __init__(self, n_dim, nodes, elements):
        self.n_dim = n_dim
        self.nodes = nodes
        self.elements = elements
        self.nodes_num, self.elements_num = len(nodes), len(elements)
        self.f = np.zeros((self.nodes_num, self.n_dim))
        self.df = np.zeros((self.nodes_num, self.n_dim))
        self.psi = np.zeros((self.nodes_num, self.n_dim))
        self.d = np.full((self.nodes_num, self.n_dim), np.nan)
        self.dd = np.full((self.nodes_num, self.n_dim), np.nan)
        self.d_free = np.isnan(self.d)
        self.stiffnessAssembling()

    # You are supposed to use this function to
    # set the force at the designated node.
    def setNodeForce(self, node_name, i_dim, f):
        self.f[node_name, i_dim] = f

    # You are supposed to use this function to
    # set the displacement at the designated node.
    def setNodeDisplacement(self, node_name, i_dim, d):
        self.d[node_name, i_dim] = d
        self.d_free[node_name, i_dim] = False

    def setDisplacement(self, boundary, i_dim, d):
        for name in boundary:
            self.d[name, i_dim] = d
            self.d_free[name, i_dim] = False

    # You are supposed to use this function to
    # set the pressure on the designated surface,
    # the surface should contain some of nodes
    def setPressure(self, boundary, pressure, coord_base):
        s = sym.symbols("s")
        N1 = (1 - s) / 2
        N2 = (1 + s) / 2
        N = sym.Matrix([N1, N2]).T
        temp = np.sqrt(1 / 3)
        N_array = np.array([N.subs([(s, -temp)]), N.subs([(s, temp)])]).reshape((2, -1))
        # 逐段边界加载荷
        for i in range(len(boundary) - 1):
            name = boundary[i:i + 2]
            # base的第一个维度是点，第二个维度是坐标
            base = self.nodes[name]
            vector = base[1] - base[0]
            length = np.sqrt(np.dot(vector, vector))
            # 高斯点计算
            gaussian_point = np.dot(N_array, base)
            # 计算高斯点处的压力值
            pressure_base = np.array([pressure.subs([(coord_base[k], gaussian_point[j, k])
                                                     for k in range(len(coord_base))])
                                      for j in range(2)])
            # 节点力计算
            # 其中length/2是雅克比阵的倒数
            f_node = (np.dot(N_array, pressure_base) * length / 2).astype(np.float32)
            # 节点力向坐标方向分解
            cs = np.flipud(abs(vector) / length)
            self.f[name] += np.einsum("i, j", f_node, cs)

    # You are supposed to use this function to
    # set the number of iterations
    def split(self, type, n_step):
        self.type = type
        self.n_step = n_step
        if type == "Force":
            self.df = self.f / n_step
            self.f = np.zeros_like(self.df)
        else:
            self.dd = self.d / n_step
            index = ~np.isnan(self.dd)
            self.d[index] = 0

    def stiffnessAssembling(self):
        self.K_global = np.zeros(shape=(self.nodes_num, self.n_dim,
                                        self.nodes_num, self.n_dim))
        for elem in self.elements:
            k_temp = elem.K_element
            for i, m in enumerate(elem.node_list):
                for j, n in enumerate(elem.node_list):
                    self.K_global[m, :, n, :] += k_temp[i, :, j, :]

    def displacementBoundaryCondition(self, u_value, flag=1):
        if flag == 0:
            f_cal = self.f - np.einsum("minj, nj->mi", self.K_global, u_value)
        else:
            f_cal = self.df - self.psi\
                    - np.einsum("minj, nj->mi", self.K_global, u_value)
        self.K_free = self.K_global[self.d_free][:, self.d_free]
        self.f_free = f_cal[self.d_free]

    def solveStiffFuction(self, u_value, flag=1):
        u_value[self.d_free] = 0
        self.displacementBoundaryCondition(u_value, flag=flag)
        u_free = np.linalg.solve(self.K_free, self.f_free)
        u = np.zeros_like(self.d_free, dtype=np.float32)
        tempPointer = 0
        for i in range(self.nodes_num):
            for j in range(self.n_dim):
                if self.d_free[i, j]:
                    u[i, j] = u_free[tempPointer]
                    tempPointer += 1
                else:
                    u[i, j] = u_value[i, j]
        return u, np.einsum('ijkl, kl->ij', self.K_global, u)

    def initStep(self):
        self.u_calculated = []
        self.f_calculated = []
        u_value = self.d.copy()
        u, f = self.solveStiffFuction(u_value, flag=0)
        self.u_calculated.append(u)
        self.f_calculated.append(f)

    def getM(self, d_e, elem):
        d_e0 = np.einsum("pii, pjk->pjk",
                         d_e, np.ones_like(d_e)) / 3
        d_ep = elem.cons.getEp(d_e)
        d_ee = d_e - d_e0 - d_ep
        d_ee_eq = np.sqrt(2 / 3 * np.einsum("pij, pij->p", d_ee, d_ee))
        alpha = 0.0002
        M = 1 + d_ee_eq / alpha
        M = M.astype(int) + 1
        return M

    def solve(self):
        V = []
        self.initStep()
        V_0 = 0
        for elem in self.elements:
            V_0 += elem.getV()
        V.append(V_0)
        print("step-0...")
        for i in range(self.n_step):
            u_value = self.dd.copy()
            d_u, d_f = self.solveStiffFuction(u_value)
            #######################################
            # 塑性迭代
            for elem in self.elements:
                d_u_elem = np.array([d_u[j] for j in elem.node_list])
                r = elem.plasticJudge(d_u_elem)
                d_e, d_s = elem.getDEAndDS(d_u_elem)
                d_s0 = np.einsum("pij,p->pij", d_s, r)
                elem.cons.updateSigma(d_s0)
                d_et = np.einsum("pij,p->pij", d_e, (1 - r))
                M = self.getM(d_e, elem)
                # 按积分点分别进行塑性迭代，调用PlasticIteration方法
                elem.cons.PlasticIteration(d_et, M)
                elem.updateElementStiffness(d_u_elem)
            self.stiffnessAssembling()
            #######################################
            self.u_calculated.append(d_u + self.u_calculated[-1])
            self.f_calculated.append(d_f + self.f_calculated[-1])
            V_0 = 0
            for elem in self.elements:
                V_0 += elem.getV()
            V.append(V_0)
            print("step-" + str(i + 1) + "...")
        return self.u_calculated, self.f_calculated, V

    def integralF(self, boundary):
        F_inte = [sum(self.f_calculated[i][boundary])
                  for  i in range(self.n_step + 1)]
        return F_inte
