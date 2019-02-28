import matplotlib.pyplot as plt
import numpy as np
import os
import Tkinter as tk
import tkFileDialog as tkf
from tqdm import tqdm
from ransac import ransac

from rkf_analysis import RkfAnalysis


class RkfMultiAnalysis(object):

    def __init__(self, rs_thresh=5):

        tkroot = tk.Tk()
        filetypes = (("ROS bag files", "*.bag"), ("all files", "*.*"))
        self.file_list = tkf.askopenfilenames(initialdir=os.getcwd(), filetypes=filetypes)
        tkroot.destroy()

        self.rs_thresh = rs_thresh
        self.rs_mdl = [0, np.inf]
        self.inliers = []

        self.cgain = np.empty((0, 2))
        self.mean_gain = np.empty((0, 2))
        self._calc_compound_gain()
        self._calc_means(robust=False)

    # Instantiate an RkfAnalysis object for each file and extract gain arrays
    def _calc_compound_gain(self, sort=True, rtype='gain'):

        for file in tqdm(self.file_list):

            rka = RkfAnalysis(file)
            self.cgain = np.concatenate((self.cgain, np.array(rka.calc_response(rka.ang_pos, rka.wing_diff, rtype=rtype))), axis=0)

        if sort:
            self.cgain = self.cgain[np.argsort(self.cgain[:, 0])]

    # Calculate the per-frequency gain means with RANSAC outlier-rejection
    def _calc_means(self, robust=True):

        self.inliers = np.array([]).astype(bool)
        if ransac:
            meanstd = lambda data: [np.mean(data), np.std(data)]
            azscore = lambda data, mdl: np.abs((data-mdl[0])/mdl[1])
            mse = lambda data, mdl: np.mean((data - mdl[0])**2)

        for freq in tqdm(np.unique(self.cgain[:, 0])):
            data = self.cgain[self.cgain[:, 0] == freq, 1]
            if robust:
                mdl = ransac(data, meanstd, azscore, mse, thresh=self.rs_thresh, max_iters=1e4)
                mean = mdl[0]
                self.inliers = np.append(self.inliers, np.abs(data-mdl[0])/mdl[1] < self.rs_thresh)
            else:
                mean = np.mean(data)
                self.inliers = np.append(self.inliers, np.ones(data.shape).astype(bool))
            self.mean_gain = np.append(self.mean_gain, [[freq, mean]], axis=0)

    # Plot compound gain array
    def plot_gain(self, normalize=True):

        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        comp = self.cgain
        mean = self.mean_gain

        # inliers = np.abs(comp[:, 1] - self.rs_mdl[0]) / self.rs_mdl[1] < self.rs_thresh
        if normalize:
            comp[:, 1] = comp[:, 1] / np.max(self.mean_gain[:, 1])
            mean[:, 1] = mean[:, 1] / np.max(self.mean_gain[:, 1])

        plt.plot(comp[self.inliers, 0], comp[self.inliers, 1], '.')
        plt.plot(comp[~self.inliers, 0], comp[~self.inliers, 1], '.', markerfacecolor='none', c=c[0])
        plt.plot(mean[:, 0], mean[:, 1], '.-', markersize=12, markerfacecolor='none', c=c[1])
        plt.ylim([-0.1, np.max(comp[self.inliers, 0])*1.1])

        plt.title('Post-RANSAC Frequency Response Curve')
        plt.legend(['Inliers', 'Outliers', 'Model mean'])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain (a.u.)')


rkm = RkfMultiAnalysis()
rkm.plot_gain()
