import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 100, 0.01)
impulse = t*0
for n in np.arange(1, 11):
    impulse += np.sin(n*t)
impulse += np.random.normal(0, 0.1, impulse.shape)

# t = np.arange(0, 100, 0.01, dtype=float)
# impulse = np.sin(np.pi*t/10)

plt.plot(t, impulse)
plt.xlim([0, 5])

freq = np.fft.fft(impulse)
plt.figure()
plt.plot(abs(freq))
