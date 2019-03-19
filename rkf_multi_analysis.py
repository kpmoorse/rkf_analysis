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

    def __init__(self, rs_thresh=5, file_list=None, nTrials=1, varnames=None, rtype='magnitude'):

        tkroot = tk.Tk()
        filetypes = (("ROS bag files", "*.bag"), ("all files", "*.*"))
        if not file_list:
            file_list = tkf.askopenfilenames(initialdir=os.getcwd(), filetypes=filetypes)
        elif (type(file_list) == str) and os.path.isdir(file_list):
            file_list = tkf.askopenfilenames(initialdir=file_list, filetypes=filetypes)
        tkroot.destroy()

        assert len(file_list) % nTrials == 0,\
            "Total number of runs must be an integer multiple of nTrials"

        self.file_list = file_list
        self.varnames = varnames
        self.varlist = []
        self.varlabels = []

        self.rtype = rtype
        self.rs_thresh = rs_thresh
        self.rs_mdl = [0, np.inf]
        self.inliers = []
        self.nTrials = nTrials

        self.cgain = np.empty((0, 2))
        self.rsq = np.empty((0, 1))
        self.stats = np.empty((0, 2))
        self._calc_compound_gain(rtype=self.rtype)
        self._calc_stats(robust=False)

    # Instantiate an RkfAnalysis object for each file and extract gain arrays
    def _calc_compound_gain(self, sort=True, rtype='magnitude'):

        for file in tqdm(self.file_list):

            rka = RkfAnalysis(file)

            if not self.varnames:
                self.varlist = (rka.ang_pos, rka.head_angle)
            else:
                self.varlist = []
                for i, var in enumerate(self.varnames):
                    self.varlist.append(getattr(rka, var))

            fit = rka.calc_sinfit(self.varlist[0], self.varlist[1], rtype=rtype)
            self.cgain = np.concatenate((self.cgain, np.array(fit[0])), axis=0)
            self.rsq = np.append(self.rsq, np.array(fit[1]))

        self.varlabels = []
        for var in self.varnames:
            self.varlabels.append(getattr(rka, var).label())

        if sort:
            self.cgain = self.cgain[np.argsort(self.cgain[:, 0])]

    # Calculate the per-frequency gain means with optional RANSAC outlier-rejection
    def _calc_stats(self, robust=True):

        self.inliers = np.array([]).astype(bool)
        self.stats = np.empty((0, 3))

        if ransac:
            meanstd = lambda data: [np.mean(data), np.std(data)]
            azscore = lambda data, mdl: np.abs((data-mdl[0])/mdl[1])
            mse = lambda data, mdl: np.mean((data - mdl[0])**2)

        # for freq in tqdm(np.unique(self.cgain[:, 0])):
        for freq in np.unique(self.cgain[:, 0]):
            rundata = self.cgain[self.cgain[:, 0] == freq, 1]
            flydata = np.mean(np.reshape(rundata, (self.nTrials, -1)), axis=0)  # Average per fly over trials
            if robust:
                mdl = ransac(flydata, meanstd, azscore, mse, thresh=self.rs_thresh, max_iters=5e4)
                mean = mdl[0]
                stderr = stats.sem(flydata[np.abs(flydata-mdl[0])/mdl[1] < self.rs_thresh])
                self.inliers = np.append(self.inliers, np.abs(rundata-mdl[0])/mdl[1] < self.rs_thresh)
            else:
                mean = np.nanmean(flydata)
                stderr = stats.sem(flydata)
                self.inliers = np.append(self.inliers, np.ones(rundata.shape).astype(bool))
            self.stats = np.append(self.stats, [[freq, mean, stderr]], axis=0)

    # Plot compound gain array
    def plot_gain(self, normalize=True, raw=True, rsq=False):

        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        comp = self.cgain.copy()
        stats = self.stats.copy()

        # inliers = np.abs(comp[:, 1] - self.rs_mdl[0]) / self.rs_mdl[1] < self.rs_thresh
        if normalize:
            norm = np.nanmax(self.stats[:, 1])
            comp[:, 1] = comp[:, 1] / norm
            stats[:, 1:] = stats[:, 1:] / norm

        if raw:
            if rsq:
                plt.scatter(comp[self.inliers, 0], comp[self.inliers, 1], c=self.rsq)
                cbar = plt.colorbar()
                cbar.set_label('Model Fit ($R^2$)')
            else:
                plt.scatter(comp[self.inliers, 0], comp[self.inliers, 1])
            plt.plot(comp[~self.inliers, 0], comp[~self.inliers, 1], '.', markerfacecolor='none', c=c[0])
        plt.errorbar(stats[:, 0], stats[:, 1], yerr=[2*x for x in stats[:, 2]],
                     fmt='.-', capsize=5, markersize=12, markerfacecolor='none', c=c[1], zorder=3)
        maxlim = np.max(comp[self.inliers, 1])
        plt.ylim([maxlim*-0.1, maxlim*1.1])

        if np.sum(self.inliers) != comp.shape[0]:
            plt.title('Post-RANSAC Frequency Response Curve')
            plt.legend(['Inliers', 'Outliers', 'Model mean'])
        else:
            plt.title('Frequency Response Curve')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(self.varlabels[1])


if __name__ == '__main__':
    rkm = RkfMultiAnalysis(nTrials=4, varnames=["ang_pos", "head_angle"], rtype='magnitude')
    rkm.plot_gain(normalize=False, raw=True)
