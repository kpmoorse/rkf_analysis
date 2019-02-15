import rosbag
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.signal as sps
import numpy as np
import os
import Tkinter as tk
import tkFileDialog as tkf
import scipy.signal as sps
from py_pll import PyPLL

# Numpy array with optional name & units metadata
class Variable(np.ndarray):

    def __new__(cls, array, dtype=None, order=None, name=None, units=None):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)
        obj.name = name
        obj.units = units
        return obj


class RkfAnalysis(object):

    # Check data integrity, initialize variables, and call preprocessing functions
    def __init__(self, bagfile=None):

        self.tkroot = tk.Tk()
        if not bagfile:
            bagfile = tkf.askopenfilename(initialdir=os.getcwd())
        elif os.path.isdir(bagfile):
            bagfile = tkf.askopenfilename(initialdir=bagfile)
        self.tkroot.destroy()

        self.bag = rosbag.Bag(bagfile, "r")
        self.bridge = CvBridge()

        # Check bag topic list
        self.topics = []
        self.msg_count = {}
        self.data_is_valid = []
        self._check_data()
        self.var_names = {}

        # Initialize data variables with metadata
        self.kf_time = Variable(np.zeros(self.msg_count["/kinefly/flystate"]),
                                name="Kinefly Time", units="sec")
        self.kf_t0 = []

        self.left_angle = Variable(self.kf_time.copy(), name="Left Wing Angle", units="deg")
        self.right_angle = Variable(self.kf_time.copy(), name="Right Wing Angle", units="deg")
        self.head_angle = Variable(self.kf_time.copy(), name="Head Angle", units="deg")
        self.ab_angle = Variable(self.kf_time.copy(), name="Abdomen Angle", units="deg")

        self.as_time = Variable(np.zeros(self.msg_count["/autostep/motion_data"]),
                                name="Autostep Time", units="sec")
        self.ang_pos = self.as_time.copy()
        self.ang_vel = self.as_time.copy()

        # Allocate and preprocess data arrays
        self._extract_data()
        self._calc_vars()
        self._resample_params()

    # Check completeness and length of bag file
    def _check_data(self):

        self.topics = self.bag.get_type_and_topic_info()[1].keys()
        self.data_is_valid = True
        for topic in ("/kinefly/flystate", "/autostep/motion_data"):
            if topic not in self.topics:
                self.data_is_valid = False
            else:
                self.msg_count[topic] = self.bag.get_message_count(topic)

    # Loop over rosbag and save messages to arrays
    def _extract_data(self):

        if not self.data_is_valid:
            raise ValueError("Required topic(s) not found in bag file")

        # Initialize counters
        ixkf = 0
        ixas = 0

        for topic, msg, t in self.bag.read_messages():

            # Extract kinefly (kf) data, filling in missing data with NaN
            if topic == "/kinefly/flystate":

                self.kf_time[ixkf] = t.to_sec()
                self.left_angle[ixkf] = msg.left.angles[0] if msg.left.angles else np.nan
                self.right_angle[ixkf] = msg.right.angles[0] if msg.right.angles else np.nan
                self.head_angle[ixkf] = msg.head.angles[0] if msg.head.angles else np.nan
                self.ab_angle[ixkf] = msg.abdomen.angles[0] if msg.abdomen.angles else np.nan

                ixkf += 1

            # Extract autostep (as) data
            if topic == "/autostep/motion_data":

                self.as_time[ixas] = t.to_sec()
                self.ang_pos[ixas] = msg.position

                ixas += 1

    # Calculate derived parameters
    def _calc_vars(self):

        self.ang_vel[:] = self.smooth_deriv(self.as_time, self.ang_pos)
        self.left_angle[:] = self.sliding_average(self.left_angle, 11)
        self.right_angle[:] = self.sliding_average(self.right_angle, 11)
        self.head_angle[:] = self.sliding_average(self.head_angle, 11)
        self.ab_angle[:] = self.sliding_average(self.ab_angle, 11)
        self.wing_diff = Variable(self.left_angle - self.right_angle, name="Wingbeat Difference (L-R)", units="deg")
        self.rng = np.bitwise_and(self.as_time[0] <= self.kf_time, self.kf_time <= self.as_time[-1])

    # Resample autostep variables to Kinefly time vector
    def _resample_params(self):

        self.ang_pos = Variable(self.resample(self.as_time, self.ang_pos, self.kf_time), name="Angular Position", units="deg")
        self.ang_vel = Variable(self.resample(self.as_time, self.ang_vel, self.kf_time), name="Angular Velocity", units="deg/sec")

    # Initialize and call cubic spline for resampling
    @staticmethod
    def resample(x1, y1, x2):

        spline = spi.CubicSpline(x1, y1)
        y2 = spline.__call__(x2, extrapolate=False)
        return y2

    # Shift an array, padding with zeros to maintain length
    @staticmethod
    def pad_shift(x, dt):

        dt = -dt
        j = dt < 1
        k = np.sign(dt)

        y = np.pad(x, abs(dt), 'edge')
        y = y[2*dt-j::k][::k]
        return y

    def parse_freq(self, x, pll_params=(0.01, 0.707, 1000), schmitt_params=None):

        pll = PyPLL(params=pll_params)
        pll.run(sps.hilbert(x))
        phase = pll.Phi  # [phi / (2*np.pi) for phi in pll.Phi]
        freq = np.diff(phase) / np.diff(self.kf_time[:2])

        if not schmitt_params:
            schmitt_params = (0., 1., 0.1)  # (offset, period, hysteresis)
        off, per, hys = schmitt_params

        freq_list = [[0, round(freq[0]/per)*per]]

        # Apply software schmitt trigger to detect line crossings
        while True:

            rail_hi = freq[freq_list[-1][0]+1:] - freq_list[-1][1] - (per / 2 + hys)
            rail_lo = freq[freq_list[-1][0]+1:] - freq_list[-1][1] + (per / 2 + hys)
            zc = np.concatenate((np.where(np.abs(np.diff(np.sign(rail_hi))))[0],
                                 np.where(np.abs(np.diff(np.sign(rail_lo))))[0]))
            if len(zc) == 0: break

            zc = int(np.min(zc) + freq_list[-1][0] + 2)

            freq_list.append([zc, round(freq[zc]/per)*per])

        return freq_list

    # Calculate cross-correlation and return centered argmax
    # *** Does not support negative correlations
    def xc_delay(self, x1, x2):

        xc = sps.correlate(self.nan_interp(x1), self.nan_interp(x2))

        # Apply Bartlett window to bias toward low absolute delays
        window = np.bartlett(len(xc))
        xc = xc * window
        xc = abs(xc * window)  # often selects the wrong peak due to noise

        m = np.argmax(xc)
        l = len(xc)
        dt = m - (l-1)/2

        return dt

    # Calculate the nth derivative and apply n+1 smoothing filters
    def smooth_deriv(self, x, y, n=1, N=5):

        deriv = y.copy()

        # Calculate nth derivative
        for i in range(n):
            deriv = (deriv[2:] - deriv[:-2]) / (x[2:] - x[:-2])
            deriv = np.pad(deriv, 1, "constant")

        # Apply n+1 smoothing filters of width N
        for i in range(n+1):
            deriv = self.sliding_average(deriv, N)

        return deriv

    # Apply a smoothing filter, padding with edge values
    def sliding_average(self, x, N):

        assert N % 2 == 1
        vec = self.nan_interp(x)
        vec = np.cumsum(vec, dtype=float)
        vec[N:] = vec[N:] - vec[:-N]
        vec = vec[N-1:] / N
        return np.pad(vec, (N-1)/2, "edge")

    # Linearly interpolate all NaN values
    def nan_interp(self, y):

        nans, x = self._nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y

    # Generate helper functions for nan_interp
    @staticmethod
    def _nan_helper(y):

        return np.isnan(y), lambda z: z.nonzero()[0]

    # Plot x1 and x2 against time
    def plot_timecourse(self, x1, x2, xcorr=True):

        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        rng = np.bitwise_and(self.as_time[0] <= self.kf_time, self.kf_time <= self.as_time[-1])
        delay = 0
        if xcorr:
            delay = self.xc_delay(x1[rng], x2[rng])
            x2 = Variable(self.pad_shift(x2.copy(), delay), name=x2.name, units=x2.units)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.kf_time - self.kf_time[0], x1)

        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('%s (%s)' % (x1.name, x1.units), color=c[0])
        for tl in ax1.get_yticklabels():
            tl.set_color(c[0])

        ax2 = ax1.twinx()
        ax2.plot(self.kf_time - self.kf_time[0], x2, c[1])
        ax2.set_ylabel('%s (%s)' % (x2.name, x2.units), color=c[1])
        for tl in ax2.get_yticklabels():
            tl.set_color(c[1])

        props = dict(boxstyle='round', facecolor='w', alpha=0.75)
        plt.text(0.05, 0.05, '$\Delta t_{corr}$ = %ims' % (1000 * delay * (self.kf_time[1] - self.kf_time[0])),
                 transform=ax1.transAxes,
                 horizontalalignment='left',
                 bbox=props)

        plt.show()

    # Plot x2 against x1
    def plot_correlation(self, x1, x2, xcorr=True):

        rng = np.bitwise_and(self.as_time[0] <= self.kf_time, self.kf_time <= self.as_time[-1])
        if xcorr:
            x2 = Variable(self.pad_shift(x2.copy(), self.xc_delay(x1[rng], x2[rng])), name=x2.name, units=x2.units)

        plt.figure()
        plt.scatter(x1[rng], x2[rng], c=rka.kf_time[rng]-rka.kf_time[rng][0])
        plt.xlabel('%s (%s)' % (x1.name, x1.units))
        plt.ylabel('%s (%s)' % (x2.name, x2.units))
        cb = plt.colorbar()
        cb.set_label("Time (sec)")

        plt.show()

    # Plot overlaid frequency responses of x1 and x2, as well as gain
    def plot_fourier(self, x1, x2, pad_factor=1):

        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # rng = np.bitwise_and(self.as_time[0] <= self.kf_time, self.kf_time <= self.as_time[-1])
        l = sum(self.rng)
        n_fft = int(2**np.ceil(np.log(l)/np.log(2))) * pad_factor

        # Calculate Fourier transforms on Bartlett-windowed data
        freq = np.fft.fftfreq(n_fft, np.diff(self.kf_time[:2]))
        sp1 = np.fft.fft(x1[self.rng][~np.isnan(x1[self.rng])] * np.bartlett(l), n=n_fft)
        sp2 = np.fft.fft((self.nan_interp(x2[self.rng]) - np.mean(x2[self.rng])) * np.bartlett(l), n=n_fft)

        gain = np.abs(sp2)/np.abs(sp1)
        lim = np.percentile(gain, 90)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(freq[:n_fft/2], np.abs(sp1)[:n_fft/2])
        ax1.set_ylabel('%s (%s)' % (x1.name, x1.units), color=c[0])
        for tl in ax1.get_yticklabels():
            tl.set_color(c[0])
        ax1.set_xlim([0, 1])

        ax2 = ax1.twinx()
        ax2.plot(freq[:n_fft/2], np.abs(sp2)[:n_fft/2], c[1])
        ax2.set_ylabel('%s (%s)' % (x2.name, x2.units), color=c[1])
        for tl in ax2.get_yticklabels():
            tl.set_color(c[1])

        ax3 = fig.add_subplot(212)
        ax3.plot(freq[:n_fft/2], gain[:n_fft/2])
        ax3.set_xlim([0, 1])
        ax3.set_ylim([-lim*0.1, lim*1.1])
        ax3.set_xlabel("Frequency (Hz)")

        ax3.set_ylabel("Gain")

    def plot_response(self):

        pad_factor = 2

        x1 = self.ang_pos[self.rng]
        x2 = self.wing_diff[self.rng]

        freq_list = self.parse_freq(x1)

        gain = []

        for i, _ in enumerate(freq_list[:-1]):

            freq1 = freq_list[i]
            freq2 = freq_list[i+1]

            x1bin = x1[freq1[0]:freq2[0]]
            x1bin -= np.mean(x1bin)
            x2bin = x2[freq1[0]:freq2[0]]
            x2bin -= np.mean(x2bin)

            l = len(x1bin)
            n_fft = int(2 ** np.ceil(np.log(l) / np.log(2))) * pad_factor

            fftfreq = np.fft.fftfreq(n_fft, np.diff(self.kf_time[:2])) * 2 * np.pi
            sp1 = np.abs(np.fft.fft(x1bin * np.bartlett(l), n=n_fft))**2
            sp2 = np.abs(np.fft.fft(x2bin * np.bartlett(l), n=n_fft))**2

            ctr = np.argmin(np.abs(fftfreq - freq1[1]))
            g_rng = np.arange(ctr - 3, ctr + 3 + 1)
            gain.append([freq1[1], np.mean(sp2[g_rng] / sp1[g_rng])])

        return gain

    # Call a common set of plot functions
    def default_plots(self, var1, var2):

        self.plot_timecourse(var1, var2, xcorr=True)
        self.plot_correlation(var1, var2, xcorr=True)
        self.plot_fourier(var1, var2, pad_factor=2)


rka = RkfAnalysis("/home/dickinsonlab/git/rkf_analysis/rosbag_data/2019-01-30/2019-01-30-17-04-00.bag")
# rka.default_plots(rka.ang_vel, rka.wing_diff)
freq_list = rka.parse_freq(rka.ang_pos[rka.rng], pll_params=(0.005, 1.5, 50000))
gain = rka.plot_response()
plt.plot([x[0] for x in freq_list], [x[1] for x in freq_list], '.')
plt.figure()
plt.plot([x[0] for x in gain], [x[1] for x in gain])
