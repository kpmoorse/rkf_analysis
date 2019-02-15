from numpy import cos, sin, angle, conj
from cmath import phase
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps


class PyPLL(object):

    def __init__(self):

        # parameters
        self.wn = 0.01  # pll bandwidth
        self.zeta = 0.707  # pll damping factor
        self.K = 1000  # pll loop gain

        # generate loop filter parameters (active PI design)
        self.t1 = self.K/(self.wn*self.wn)   # tau_1
        self.t2 = 2*self.zeta/self.wn   # tau_2

        # feed-forward coefficients (numerator)
        self.b0 = (4*self.K/self.t1)*(1.+self.t2/2.0)
        self.b1 = (8*self.K/self.t1)
        self.b2 = (4*self.K/self.t1)*(1.-self.t2/2.0)

        # feed-back coefficients (denominator)
        #    a0 =  1.0  is implied
        self.a1 = -2.0
        self.a2 = 1.0

        # filter buffer
        self.v0 = 0.0
        self.v1 = 0.0
        self.v2 = 0.0

        # Initialize states
        self.phi_hat = 0.0  # PLL's initial phase

        self.X = []
        self.Y = []
        self.y = []
        self.delta_phi = []

    def step(self, x):

        # compute PLL output from phase estimate
        self.y = cos(self.phi_hat) + 1j * sin(self.phi_hat)

        # compute error estimate
        self.delta_phi = angle(x * conj(self.y))

        # Store input and output values
        self.X.append(x.real)
        self.Y.append(self.y.real)

        # advance buffer
        self.v2 = self.v1  # shift center register to upper register
        self.v1 = self.v0  # shift lower register to center register

        # compute new lower register
        self.v0 = self.delta_phi - self.v1 * self.a1 - self.v2 * self.a2

        # compute new output
        self.phi_hat = self.v0 * self.b0 + self.v1 * self.b1 + self.v2 * self.b2


if __name__ == '__main__':

    pll = PyPLL()

    n = 400
    Phi = np.arange(n) * 0.3
    X0 = sps.hilbert(5 * cos(Phi))
    for i in range(n):
        x = X0[i]
        pll.step(x)

    plt.plot(pll.X)
    plt.plot(pll.Y)
