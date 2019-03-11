from trajectory import Trajectory
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# def Hermite(n, x):
#
#     pass
#
# def HG(x, y, (n, m), w0=1):
#
#     pass
#
# x = np.arange(-3, 3, 0.1)
# y = np.arange(-3, 3, 0.1)
# X, Y, = np.meshgrid(x, y)

t = np.arange(0, 3, 0.01)
x1 = t*0
x2 = x1.copy()
for i in range(10):
    x1 += np.sin(2 * np.pi * i * t)
    x2 += np.sin(2 * np.pi * i * t + np.random.rand()*np.pi*2)

plt.plot(t, x1, t, x2)

# while True:
#
#     freq_list = np.arange(10)
#     x = t.copy()*0
#     for freq in freq_list:
#         rnd1 = (np.random.rand(1) * 2 - 1) * 0.02
#         rnd2 = (np.random.rand(1) * 2 - 1) * 1
#         x += np.sin((freq*(1+rnd1))*t + rnd2)
#
#     l = len(x)
#     n_fft = int(2 ** np.ceil(np.log(l) / np.log(2))) * 2
#     sp = np.abs(np.fft.fft(x, n_fft))**2
#
#     plt.clf()
#     plt.subplot(211)
#     plt.plot(t, x)
#     plt.subplot(212)
#     plt.plot(sp)
#     plt.xlim((0, 100))
#
#     plt.pause(1)
