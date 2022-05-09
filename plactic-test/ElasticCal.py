import numpy as np

def getDeviatonicStress(sigma_p):
    n_dim = np.shape(sigma_p)[0]
    delta_ij = np.eye(n_dim)
    sigma_0 = np.einsum("ii", sigma_p) / n_dim
    sij = sigma_p - sigma_0 * delta_ij
    return sij

def getJ2(sigma_p):
    sij = getDeviatonicStress(sigma_p)
    J2 = np.einsum("ij,ij", sij, sij) / 2
    return J2

def getMisesStress(sigma_p):
    n_dim = np.shape(sigma_p)[0]
    if n_dim == 2:
        sigma_mod = np.zeros((3, 3))
        sigma_mod[0:2, 0:2] = sigma_p
    else:
        sigma_mod = sigma_p
    J2 = getJ2(sigma_mod)
    mises_stress = np.sqrt(3 * J2)
    return mises_stress