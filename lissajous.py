import numpy as np
import matplotlib.pyplot as plt


def lissajous(t, params):

    a, b, w, d = params

    x = lambda t: a * np.sin(w * t + d)
    y = lambda t: b * np.sin(t)

    plt.plot(x(t), y(t))
    plt.axes().set_aspect('equal', 'datalim')

lissajous(np.arange(0, 10, 0.1), (2, 1, 1, np.pi*0.9))
