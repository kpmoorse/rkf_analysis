#!/usr/bin/python
import numpy as np
from numpy import sqrt, pi, cos, exp
import matplotlib.pyplot as plt

dt = 0.005
e = dt/10
T = 10
f = lambda x, v1, v2: 1/(2*pi*1j*(x+e)) * (exp(2*pi*1j*(x+e)*v1) - exp(2*pi*1j*(x+e)*v2))
gauss = lambda x, s: 1/(s*sqrt(2*pi)) * exp(-x**2/(2*s**2))
t = np.arange(0, T, dt)
t_gap = lambda l, d: np.concatenate((np.arange(-(l+d)/2, -d/2, dt), np.arange(d/2, (l+d)/2, dt)))
t2 = t_gap(T, 0.1)
# t2 = np.arange(1, 11, dt)
N = len(t)


def lognorm(y, a=0, A=None):

    y = np.array(y).astype(float)
    b = max(np.abs(y))
    if not A:
        A = b

    vec = y.copy() / b
    lg = np.array(np.abs(vec) > a).astype(bool)
    vec[lg] = np.sign(y[lg]) * (a + np.log(abs(vec[lg] - a) + 1))
    vec = A/max(abs(vec)) * vec

    return vec


# s1 = np.sinc((t - 5)*2)
# s2 = np.sinc(t - 5)
# g1 = gauss(t-5, 1)
# g2 = gauss(t-5, 0.25)
# plt.plot(t, g1)
rep = 1
f1 = f(t-T/2, 1, 3)
f2 = lognorm(f(t-T/2, 1, 3), A=10)

f1 = np.tile(f1, rep)
f2 = np.tile(f2, rep)
t = np.arange(0, T*rep, dt)
N = len(t)
# f2 = loglim(f1.copy(), 0.5)*10
# f1_int = np.cumsum(f1) * dt

F1 = np.fft.fft(f1 * np.bartlett(N))
F2 = np.fft.fft(f2 * np.bartlett(N))
# S1 = np.fft.fft(s1 * np.bartlett(N))
# S2 = np.fft.fft(s2 * np.bartlett(N))
freq = np.fft.fftfreq(N, dt)

plt.subplot(2, 1, 1)
plt.plot(t, f1, t, f2)
# plt.plot(t, f1_int * (max(f1) / max(f1_int)))
plt.subplot(2, 1, 2)
plt.plot(freq[:N/2], (np.abs(F1)**2)[:N/2])
plt.plot(freq[:N/2], (np.abs(F2)**2)[:N/2])
plt.xlim(0, 5)
