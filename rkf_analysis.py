import rosbag
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.signal as sps
import numpy as np


# Numpy array with optional metadata
class Variable(np.ndarray):

    def __new__(cls, array, dtype=None, order=None, name=None, units=None):
        obj = np.asarray(array, dtype=dtype, order=order).view(cls)
        obj.name = name
        obj.units = units
        return obj


class RkfAnalysis(object):

    def __init__(self, bagfile):

        # Load bag file
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
        self._calc_params()
        self._resample_params()

    def _check_data(self):

        self.topics = self.bag.get_type_and_topic_info()[1].keys()
        self.data_is_valid = True
        for topic in ("/kinefly/flystate", "/autostep/motion_data"):
            if topic not in self.topics:
                self.data_is_valid = False
            else:
                self.msg_count[topic] = self.bag.get_message_count(topic)

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

    def _calc_params(self):

        self.ang_vel[:] = self.smooth_deriv(self.as_time, self.ang_pos)
        self.left_angle[:] = self.sliding_average(self.left_angle, 11)
        self.right_angle[:] = self.sliding_average(self.right_angle, 11)
        self.head_angle[:] = self.sliding_average(self.head_angle, 11)
        self.ab_angle[:] = self.sliding_average(self.ab_angle, 11)
        self.wing_diff = Variable(self.left_angle - self.right_angle, name="Wingbeat Difference (L-R)", units="deg")

    def _resample_params(self):

        self.ang_pos = Variable(self.resample(self.as_time, self.ang_pos, self.kf_time), name="Angular Position", units="deg")
        self.ang_vel = Variable(self.resample(self.as_time, self.ang_vel, self.kf_time), name="Angular Velocity", units="deg/sec")

    @staticmethod
    def resample(x1, y1, x2):

        spline = spi.CubicSpline(x1, y1)
        y2 = spline.__call__(x2, extrapolate=False)
        return y2

    @staticmethod
    def pad_shift(x, dt):

        dt = -dt
        j = dt < 1
        k = np.sign(dt)

        y = np.pad(x, abs(dt), 'edge')
        y = y[2*dt-j::k][::k]
        return y

    def xc_delay(self, x1, x2):

        xc = sps.correlate(self.nan_interp(x1), self.nan_interp(x2))
        envelope = np.arange(len(xc))
        envelope = 1 - np.abs(envelope/envelope[-1] - 0.5)
        xc = xc * envelope

        m = np.argmax(xc)
        l = len(xc)
        dt = m - (l-1)/2

        return dt

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

    def sliding_average(self, x, N):

        assert N % 2 == 1
        vec = self.nan_interp(x)
        vec = np.cumsum(vec, dtype=float)
        vec[N:] = vec[N:] - vec[:-N]
        vec = vec[N-1:] / N
        return np.pad(vec, (N-1)/2, "edge")

    def nan_interp(self, y):

        nans, x = self._nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y

    @staticmethod
    def _nan_helper(y):

        return np.isnan(y), lambda z: z.nonzero()[0]

    def plot_timecourse(self, x1, x2, xcorr=True):

        rng = np.bitwise_and(self.as_time[0] <= self.kf_time, self.kf_time <= self.as_time[-1])
        if xcorr:
            x2 = Variable(self.pad_shift(x2.copy(), self.xc_delay(x1[rng], x2[rng])), name=x2.name, units=x2.units)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.kf_time - self.kf_time[0], x1)

        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('%s (%s)' % (x1.name, x1.units))

        ax2 = ax1.twinx()
        ax2.plot(self.kf_time - self.kf_time[0], x2, 'r-')
        ax2.set_ylabel('%s (%s)' % (x2.name, x2.units), color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        plt.show()

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

    def plot_fourier(self, x1, x2):

        rng = np.bitwise_and(self.as_time[0] <= self.kf_time, self.kf_time <= self.as_time[-1])
        n_fft = int(2**(np.ceil(np.log(sum(rng)/np.log(2)))))

        freq = np.fft.fftfreq(n_fft, np.diff(self.kf_time[:2]))
        sp1 = np.fft.fft(x1[rng][~np.isnan(x1[rng])], n=n_fft)
        sp2 = np.fft.fft(self.nan_interp(x2[rng]) - np.mean(x2[rng]), n=n_fft)
        gain = np.abs(sp2)/np.abs(sp1)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(freq[:n_fft/2], np.abs(sp1)[:n_fft/2])
        ax1.set_xlim([0, 5])

        ax2 = ax1.twinx()
        ax2.plot(np.nan, '-', label='temp')
        ax2.plot(freq[:n_fft/2], np.abs(sp2)[:n_fft/2], 'r-')

        ax2.legend(['%s (%s)' % (x1.name, x1.units), '%s (%s)' % (x2.name, x2.units)])

        ax3 = fig.add_subplot(212)
        ax3.plot(freq[:n_fft/2], gain[:n_fft/2])
        ax3.set_xlim([0, 5])
        ax3.set_xlabel("Frequency (Hz)")

        ax3.legend(["Gain"])

    def default_plots(self, var1, var2, xcorr=True):

        self.plot_timecourse(var1, var2, xcorr=xcorr)
        self.plot_correlation(var1, var2, xcorr=xcorr)
        self.plot_fourier(var1, var2)


rka = RkfAnalysis("/home/dickinsonlab/git/rkf_analysis/rosbag_data/2019-02-01-15-20-31.bag")
rka.default_plots(rka.ang_vel, rka.wing_diff, xcorr=True)
