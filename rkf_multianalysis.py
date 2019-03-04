import matplotlib.pyplot as plt
import numpy as np
import os
import Tkinter as tk
import tkFileDialog as tkf
from tqdm import tqdm
from ransac import ransac
from scipy import stats

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
        self.stats = np.empty((0, 2))
        self._calc_compound_gain()
        self._calc_stats(robust=False)

    # Instantiate an RkfAnalysis object for each file and extract gain arrays
    def _calc_compound_gain(self, sort=True, rtype='gain'):

        for file in tqdm(self.file_list):

            rka = RkfAnalysis(file)
            # self.cgain = np.concatenate((self.cgain, np.array(rka.calc_response(rka.ang_pos, rka.wing_diff, rtype=rtype))), axis=0)
            self.cgain = np.concatenate((self.cgain, np.array(rka.calc_sinfit(rka.ang_pos, rka.head_angle, rtype=rtype))), axis=0)

        if sort:
            self.cgain = self.cgain[np.argsort(self.cgain[:, 0])]

    # Calculate the per-frequency gain means with RANSAC outlier-rejection
    def _calc_stats(self, robust=True):

        self.inliers = np.array([]).astype(bool)
        self.stats = np.empty((0, 3))

        if ransac:
            meanstd = lambda data: [np.mean(data), np.std(data)]
            azscore = lambda data, mdl: np.abs((data-mdl[0])/mdl[1])
            mse = lambda data, mdl: np.mean((data - mdl[0])**2)

        for freq in tqdm(np.unique(self.cgain[:, 0])):
            data = self.cgain[self.cgain[:, 0] == freq, 1]
            if robust:
                mdl = ransac(data, meanstd, azscore, mse, thresh=self.rs_thresh, max_iters=5e4)
                mean = mdl[0]
                stderr = stats.sem(data[np.abs(data-mdl[0])/mdl[1] < self.rs_thresh])
                self.inliers = np.append(self.inliers, np.abs(data-mdl[0])/mdl[1] < self.rs_thresh)
            else:
                mean = np.nanmean(data)
                stderr = stats.sem(data)
                self.inliers = np.append(self.inliers, np.ones(data.shape).astype(bool))
            self.stats = np.append(self.stats, [[freq, mean, stderr]], axis=0)

    # Plot compound gain array
    def plot_gain(self, normalize=True, raw=True):

        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        comp = self.cgain.copy()
        stats = self.stats.copy()

        # inliers = np.abs(comp[:, 1] - self.rs_mdl[0]) / self.rs_mdl[1] < self.rs_thresh
        if normalize:
            norm = np.nanmax(self.stats[:, 1])
            comp[:, 1] = comp[:, 1] / norm
            stats[:, 1:] = stats[:, 1:] / norm

        if raw:
            plt.plot(comp[self.inliers, 0], comp[self.inliers, 1], '.')
            plt.plot(comp[~self.inliers, 0], comp[~self.inliers, 1], '.', markerfacecolor='none', c=c[0])
        plt.errorbar(stats[:, 0], stats[:, 1], yerr=stats[:, 2],
                     fmt='.-', capsize=5, markersize=12, markerfacecolor='none', c=c[1])
        maxlim = np.max(comp[self.inliers, 1])
        plt.ylim([maxlim*-0.1, maxlim*1.1])

        if np.sum(self.inliers) != comp.shape[0]:
            plt.title('Post-RANSAC Frequency Response Curve')
            plt.legend(['Inliers', 'Outliers', 'Model mean'])
        else:
            plt.title('Frequency Response Curve')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain (a.u.)')


rkm = RkfMultiAnalysis()
rkm.plot_gain(normalize=False, raw=False)
