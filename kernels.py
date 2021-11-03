#print(u.reshape(-1,1))
import numpy as np
class Kernels():
    # a > 0
    def NewtonianPotential(self, r, a = 1):
        return 1.0 / (r + a)

    #a > 0 and n > 0
    def NewtonianPotentialM(self, r, a = 1, n =1):
        return 1.0 / (r**n + a)

    def Epanechnikov(self, r):
        return 0.75 * (1 - r**2)

    def Quarter(self, r):
        return 0.9375 * (1 - r**2)**2

    def Triangular(self, r):
        return 1 - np.abs(r)

    def Gaussian(self, r):
        return (2.0 * np.pi)**(-0.5) * np.exp(-0.5 * r**2)

    def Rect(self, r):
        return np.full(r.shape, 0.5)
