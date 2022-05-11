import numpy as np
import ElasticCal

class Constitutive(object):
    def __init__(self, n_gauss, n_dim, E, mu):
        self.n_gauss = n_gauss
        self.n_dim = n_dim
        self.E = E
        self.mu = mu
        lam = E * mu / (1 + mu) / (1 - 2 * mu)
        G = 0.5 * E / (1 + mu)
        self.D = np.zeros(shape=(self.n_gauss, self.n_dim, self.n_dim,
                                 self.n_dim, self.n_dim))
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                self.D[:, i, j, i, j] = self.D[:, i, j, j, i] = G
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                self.D[:, i, i, j, j] = lam
            self.D[:, i, i, i, i] = lam + 2 * G
        self.Dp = np.zeros_like(self.D)
        self.Dep = self.D - self.Dp
        self.sigma = np.zeros(shape=(self.n_gauss, self.n_dim, self.n_dim))
        self.sum_ep_eq = np.zeros(self.n_gauss)

    def yieldFunction(self, sigma):
        pass

    def yield_diff(self, sigma):
        pass

    def potentialFunction(self, sigma):
        g_func = self.yieldFunction(sigma)
        return g_func

    def potential_diff(self, sigma):
        g_diff = self.yield_diff(sigma)
        return g_diff

    def hardeningFunction(self):
        pass

    def hardening_diff(self):
        pass

    def getA(self, sigma):
        h_diff = self.hardening_diff()
        g_diff = self.potential_diff(sigma)
        A = -h_diff * np.sqrt(2 / 3 * np.einsum("pij,pij->p", g_diff, g_diff))
        return A

    def getdLam(self, d_e, sigma):
        f_diff = self.yield_diff(sigma)
        g_diff = self.potential_diff(sigma)
        A = self.getA(sigma)
        d_lam = np.einsum("pij, pijkl, pkl->p", f_diff, self.D, d_e) / \
                (np.einsum("pij, pijkl, pkl->p", f_diff, self.D, g_diff) + A)
        return d_lam

    def getEp(self, d_e):
        g_diff = self.potential_diff(self.sigma)
        d_lam = self.getdLam(d_e, self.sigma)
        d_ep = np.einsum("p,pij->pij", d_lam, g_diff)
        return d_ep

    def updateDep(self):
        f_diff = self.yield_diff(self.sigma)
        g_diff = self.potential_diff(self.sigma)
        A = self.getA(self.sigma)
        H = np.einsum("pij, pijkl, pkl->p", f_diff, self.D, g_diff) + A
        self.Dp = np.einsum("pmnkl, pkl, pij, pijrs, p->pmnrs",
                            self.D, g_diff, f_diff, self.D, 1 / H)
        self.Dep = self.D - self.Dp

    def calculateR(self, d_sigma):
        # TODO 这个R的计算方法不怎么合理
        tole = 1e-7
        sigma = self.sigma + d_sigma
        r = []
        Fm_1 = self.yieldFunction(sigma)
        Fm_0 = self.yieldFunction(self.sigma)
        for i in range(self.n_gauss):
            if Fm_1[i] > tole:
                if abs(Fm_0[i]) < tole:
                    r.append(0)
                else:
                    if Fm_0[i] != Fm_1[i]:
                        r_temp = -Fm_0[i] / (Fm_1[i] - Fm_0[i])
                        if r_temp > 1:
                            r.append(1)
                        elif r_temp < 0:
                            r.append(0)
                        else:
                            r.append(-Fm_0[i] / (Fm_1[i] - Fm_0[i]))
                    else:
                        r.append(1)
            else:
                r.append(1)
        return np.array(r)

    def updateSigma(self, d_s):
        self.sigma += d_s

    def PlasticIteration(self, d_et, M):
        d_d_e = np.einsum("pij,p->pij", d_et, 1 / M)
        for i in range(self.n_gauss):
            if np.any(d_d_e[i]):
                for j in range(M[i]):
                    # 迭代过程中实时更新应力值
                    d_d_s = np.einsum("ijkl, kl->ij", self.Dep[i], d_d_e[i])
                    self.sigma[i] += d_d_s
                    # 迭代过程中根据硬化关系实时更新内变量
                    d_d_ep = self.getEp(d_d_e)
                    d_ep_eq = np.sqrt(2/3 * np.einsum("pij,pij->p", d_d_ep, d_d_ep))
                    self.sum_ep_eq += d_ep_eq
                    self.hardeningFunction()
                    # 迭代过程中实时更新本构阵
                    self.updateDep()

class VonMises(Constitutive):
    def __init__(self, n_num, n_dim, E, mu, sigma_s, Et):
        super().__init__(n_num, n_dim, E, mu)
        self.Et = Et
        self.sigma_s0 = sigma_s
        self.k = np.full((self.n_gauss), self.sigma_s0)

    def yieldFunction(self, sigma):
        f = np.array([ElasticCal.getMisesStress(sigma_i)
                      for sigma_i in sigma])
        f_func = f - self.k
        return f_func

    def yield_diff(self, sigma):
        f_diff = np.array([ElasticCal.getDeviatonicStress(sigma_i)
                           for sigma_i in sigma])
        J2 = np.einsum("pij,pij->p", f_diff, f_diff) / 2
        f_diff = np.einsum("pij,p->pij", f_diff, 1 / np.sqrt(J2)) * np.sqrt(3) / 2
        return f_diff

    # TODO 这个硬化函数随便取的，看上去多半还有问题
    def hardeningFunction(self):
        Ep = self.E * self.Et / (self.E - self.Et)
        self.k = Ep * self.sum_ep_eq + self.sigma_s0
        return self.k

    def hardening_diff(self):
        Ep = self.E * self.Et / (self.E - self.Et)
        h_diff = Ep
        return h_diff

