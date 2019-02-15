from trajectory import Trajectory
import numpy as np
import matplotlib.pyplot as plt

# freq_list = [[10, 1], [10, 2], [10, 3], [10, 4], 10]
tt = 10
freq_list = [[tt, (i+1)*0.5] for i in range(10)]
dt = 0.01

# Calculate fft for entire timecourse
tau = sum([freq[0] for freq in freq_list])
num_pts = int(tau/dt)
t = dt*np.arange(num_pts)

trj = Trajectory(t)
trj.set_frequency(trj.stepwise(freq_list))

pad_factor = 2
n_fft = int(2**np.ceil(np.log(num_pts)/np.log(2))) * pad_factor
F = np.fft.fft(trj.position * np.hanning(num_pts), n=n_fft)
freq = np.fft.fftfreq(n_fft, dt)

plt.plot(freq[:n_fft/2], (np.abs(F)**2)[:n_fft/2])

# Repeat with separate trials
tau = tt
num_pts = int(tau/dt)
t = dt*np.arange(num_pts)

pad_factor = 8
n_fft = int(2**np.ceil(np.log(num_pts)/np.log(2))) * pad_factor
freq = np.fft.fftfreq(n_fft, dt)

Fi = np.zeros(n_fft)
for i in range(len(freq_list)):
    Fi = Fi + np.fft.fft(trj.position[i*int(tt/dt):(i+1)*int(tt/dt)] * np.hanning(num_pts), n=n_fft)
plt.plot(freq[:n_fft/2], (np.abs(Fi)**2)[:n_fft/2])
