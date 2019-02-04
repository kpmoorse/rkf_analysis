import rosbag
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.signal as sps
import numpy as np


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

        # Initialize data variables
        self.kf_time = np.array([])
        self.kf_t0 = []

        self.left_angle = np.array([])
        self.right_angle = np.array([])
        self.head_angle = np.array([])
        self.ab_angle = np.array([])

        self.as_time = np.array([])
        self.ang_pos = np.array([])

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

        # Preallocate data arrays
        self.kf_time = np.zeros(self.msg_count["/kinefly/flystate"])
        self.left_angle = self.kf_time.copy()
        self.right_angle = self.kf_time.copy()
        self.head_angle = self.kf_time.copy()
        self.ab_angle = self.kf_time.copy()

        self.as_time = np.zeros(self.msg_count["/autostep/motion_data"])
        self.ang_pos = self.as_time.copy()

        # Initialize counters
        ixkf = 0
        ixas = 0

        for topic, msg, t in self.bag.read_messages():

            # Extract kinefly data, ignoring missing data points
            if topic == "/kinefly/flystate":

                self.kf_time[ixkf] = t.to_sec()
                self.left_angle[ixkf] = msg.left.angles[0] if msg.left.angles else np.nan
                self.right_angle[ixkf] = msg.right.angles[0] if msg.right.angles else np.nan
                self.head_angle[ixkf] = msg.head.angles[0] if msg.head.angles else np.nan
                self.ab_angle[ixkf] = msg.abdomen.angles[0] if msg.abdomen.angles else np.nan

                ixkf += 1

            # Extract autostep data
            if topic == "/autostep/motion_data":

                self.as_time[ixas] = t.to_sec()
                self.ang_pos[ixas] = msg.position

                ixas += 1

    def _calc_params(self):

        self.ang_vel = self.smooth_deriv(self.as_time, self.ang_pos)
        self.wing_diff = self.left_angle - self.right_angle

    def _resample_params(self):

        self.ang_pos = self.resample(self.as_time, self.ang_pos, self.kf_time)
        self.ang_vel = self.resample(self.as_time, self.ang_vel, self.kf_time)

    # def _create_var_dict(self):
    #
    #     self.var_names = {"Kinefly Time": [self.kf_time, "deg"],
    #                       "Left Wing Angle": [self.left_angle, "deg"],
    #                       "Right Wing Angle": [self.right_angle, "deg"],
    #                       "Head Angle": [self.head_angle, "deg"],
    #                       "Abdomen Angle": [self.ab_angle, "deg"],
    #                       "Wingbeat Difference": [self.wing_diff, "deg"],
    #                       "Autostep Time": [self.as_time, "sec"],
    #                       "Angular Position": [self.ang_pos, "deg"],
    #                       "Angular Velocity": [self.ang_vel, "deg/s"]
    #                       }

    @staticmethod
    def resample(x1, y1, x2):

        spline = spi.CubicSpline(x1, y1)
        y2 = spline.__call__(x2, extrapolate=False)
        return y2

    def pad_shift(self, x, dt):

        dt = -dt
        j = dt < 1
        k = np.sign(dt)

        y = np.pad(x, abs(dt), 'edge')
        y = y[2*dt-j::k][::k]
        return y

    def xc_delay(self, x1, x2):

        xc = sps.correlate(self.nan_interp(x1), self.nan_interp(x2))
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

    def plot_timecourse(self, x1, x2, xc=True):

        if xc:
            x2 = self.pad_shift(x2, self.xc_delay(x1, x2))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.kf_time - self.kf_time[0], x1)

        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Angular Velocity (deg/s)')

        ax2 = ax1.twinx()
        ax2.plot(self.kf_time - self.kf_time[0], x2, 'r-')
        ax2.set_ylabel('Wingbeat Difference (L-R)', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        # x2b = self.pad_shift(x2, self.xc_delay(x1, x2))
        # ax2.plot(self.kf_time - self.kf_time[0], x2b, 'g-')

        plt.show()

    def plot_correlation(self, x1, x2, xc=True):

        if xc:
            x2 = self.pad_shift(x2, self.xc_delay(x1, x2))

        rng = np.bitwise_and(self.as_time[0] <= self.kf_time, self.kf_time <= self.as_time[-1])

        plt.figure()
        plt.scatter(x1[rng], x2[rng], c=rka.kf_time[rng]-rka.kf_time[rng][0])
        plt.xlabel('Angular Velocity (deg/s)')
        plt.ylabel('Wingbeat Difference (L-R)')
        cb = plt.colorbar()
        cb.set_label("Time (sec)")

        plt.show()

    def default_plots(self):

        self.plot_timecourse(x1=self.ang_vel, x2=self.wing_diff)
        self.plot_correlation(x1=self.ang_vel, x2=self.wing_diff)


rka = RkfAnalysis("/home/dickinsonlab/git/rkf_analysis/rosbag_data/2019-02-01-15-17-24.bag")
rka.default_plots()
