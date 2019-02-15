from numpy import cos, sin, angle, conj
from cmath import phase
import numpy as np
import matplotlib.pyplot as plt


def main():

    # parameters
    phase_offset = 0.00  # carrier phase offset
    frequency_offset = 0.30  # carrier frequency offset
    wn = 0.01  # pll bandwidth
    zeta = 0.707  # pll damping factor
    K = 1000  # pll loop gain
    n = 400  # number of samples

    # generate loop filter parameters (active PI design)
    t1 = K/(wn*wn)   # tau_1
    t2 = 2*zeta/wn   # tau_2

    # feed-forward coefficients (numerator)
    b0 = (4*K/t1)*(1.+t2/2.0)
    b1 = (8*K/t1)
    b2 = (4*K/t1)*(1.-t2/2.0)

    # feed-back coefficients (denominator)
    #    a0 =  1.0  is implied
    a1 = -2.0
    a2 = 1.0

    # print filter coefficients (as comments)
    # printf("#  b = [b0:%12.8f, b1:%12.8f, b2:%12.8f]\n", b0, b1, b2)
    # printf("#  a = [a0:%12.8f, a1:%12.8f, a2:%12.8f]\n", 1., a1, a2)

    # filter buffer
    v0 = 0.0
    v1 = 0.0
    v2 = 0.0

    # initialize states
    phi = phase_offset  # input signal's initial phase
    phi_hat = 0.0  # PLL's initial phase

    # # print line legend to standard output
    # printf("# %6s %12s %12s %12s %12s %12s\n",
    #         "index", "real(x)", "imag(x)", "real(y)", "imag(y)", "error")

    # run basic simulation
    X = []
    Y = []
    for i in range(n):
        # compute input sinusoid and update phase
        x = cos(phi) + 1j*sin(phi)
        phi += frequency_offset

        # compute PLL output from phase estimate
        y = cos(phi_hat) + 1j*sin(phi_hat)

        # compute error estimate
        delta_phi = angle(x * conj(y))

        # print results to standard output
        print("  %6u %12.8f %12.8f %12.8f %12.8f %12.8f\n",
                  i, x, y, delta_phi)
        X.append(x.real)
        Y.append(y.real)
        # push result through loop filter, updating phase estimate

        # advance buffer
        v2 = v1  # shift center register to upper register
        v1 = v0  # shift lower register to center register

        # compute new lower register
        v0 = delta_phi - v1*a1 - v2*a2

        # compute new output
        phi_hat = v0*b0 + v1*b1 + v2*b2

    return X, Y

X, Y = main()
plt.plot(X)
plt.plot(Y)

# class PyPll(object):
#
#     def __init__(self):
#
#         # parameters
#         self.wn = 0.01  # pll bandwidth
#         self.zeta = 0.707  # pll damping factor
#         self.K = 1000  # pll loop gain
#         self.n = 400  # number of samples
#
#         # self.x = []
#         self.y = []
#         # self.X = []  # np.zeros(self.n, dtype=complex)
#         # self.Y = []  # np.zeros(self.n, dtype=complex)
#         self.delta_phi = []
#
#         # generate loop filter parameters (active PI design)
#         self.t1 = self.K / (self.wn * self.wn)  # tau_1
#         self.t2 = 2 * self.zeta / self.wn  # tau_2
#
#         # feed-forward coefficients (numerator)
#         self.b0 = (4 * self.K / self.t1) * (1. + self.t2 / 2.0)
#         self.b1 = (8 * self.K / self.t1)
#         self.b2 = (4 * self.K / self.t1) * (1. - self.t2 / 2.0)
#
#         # feed-back coefficients (denominator)
#         # a0 =  1.0  is implied
#         self.a1 = -2.0
#         self.a2 = 1.0
#
#         # filter buffer
#         self.v0 = 0.0
#         self.v1 = 0.0
#         self.v2 = 0.0
#
#         # initialize states
#         self.phi_hat = 0.0  # PLL's initial phase
#
#     def step(self, x):
#
#         # compute PLL output from phase estimate
#         self.y = cos(self.phi_hat) + 1j * sin(self.phi_hat)
#
#         # compute error estimate
#         self.delta_phi = phase(x * self.y.conjugate())
#
#         # compute new lower register
#         self.v0 = self.delta_phi - self.v1 * self.a1 - self.v2 * self.a2
#
#         # compute new output
#         self.phi_hat = self.v0 * self.b0 + self.v1 * self.b1 + self.v2 * self.b2
#
#
# pll = PyPll()
#
# phase_offset = 0.00
# frequency_offset = 0.30
# phi = phase_offset
#
# X = []
# Y = []
# for i in range(400):
#     x = cos(phi) + 1j * sin(phi)
#     phi += frequency_offset
#     pll.step(x)
#     X.append(x.real)
#     Y.append(pll.y.real)
#
# plt.plot(X)
# plt.plot(Y)