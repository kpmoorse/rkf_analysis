from numpy import cos, sin, angle, conj
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sps


class PyPLL(object):

    def __init__(self, params=(0.01, 0.707, 1000)):

        self.wn = []
        self.zeta = []
        self.K = []

        self.X = []
        self.Y = []
        self.Phi = []

        self.y = []
        self.delta_phi = []

        self.set_params(params)

    # Initialize primary params
    def set_params(self, params):

        # parameters
        self.wn = params[0]  # pll bandwidth
        self.zeta = params[1]  # pll damping factor
        self.K = params[2]  # pll loop gain
        self._calc_params()

    # Calculate secondary params
    def _calc_params(self):

        # generate loop filter parameters (active PI design)
        self.t1 = self.K / (self.wn * self.wn)   # tau_1
        self.t2 = 2 * self.zeta / self.wn   # tau_2

        # feed-forward coefficients (numerator)
        self.b0 = (4 * self.K / self.t1)*(1. + self.t2 / 2.0)
        self.b1 = (8 * self.K / self.t1)
        self.b2 = (4 * self.K / self.t1)*(1. - self.t2 / 2.0)

        # feed-back coefficients (denominator)
        #    a0 =  1.0  is implied
        self.a1 = -2.0
        self.a2 = 1.0

        # filter buffer
        self.v0 = 0.0
        self.v1 = 0.0
        self.v2 = 0.0

        # Initialize state
        self.phi_hat = 0.0  # PLL's initial phase

    # Step through a whole vector of input data
    def run(self, X):

        for x in X:
            self.step(x)

    # Step forward in response to new input data
    # Input data must be complex; real data should be preprocessed with scipy.signal.hilbert()
    def step(self, x):

        # compute PLL output from phase estimate
        self.y = cos(self.phi_hat) + 1j * sin(self.phi_hat)

        # compute error estimate
        self.delta_phi = angle(x * conj(self.y))

        # Store input and output values
        self.X.append(x.real)
        self.Y.append(self.y.real)
        self.Phi.append(self.phi_hat)

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
    pll.run(X0)

    plt.plot(pll.X)
    plt.plot(pll.Y)
