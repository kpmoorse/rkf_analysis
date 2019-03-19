import Tkinter as tk
import tkFileDialog as tkf
import os
import matplotlib.pyplot as plt

from rkf_multi_analysis import RkfMultiAnalysis


class RkfMetaAnalysis(object):

    def __init__(self, initialdir=None, nCats=2, varnames=None):

        tkroot = tk.Tk()
        filetypes = (("ROS bag files", "*.bag"), ("all files", "*.*"))
        if not initialdir:
            initialdir = os.getcwd()

        self.meta_list = []
        for i in range(nCats):
            self.meta_list.append(tkf.askopenfilenames(initialdir=initialdir, filetypes=filetypes,
                                  title='Select files for category #%i' % (i+1)))
        tkroot.destroy()

        self.varnames = varnames
        self._extract_data()

    def _extract_data(self):

        self.meta_stats = []
        for file_list in self.meta_list:

            rkm = RkfMultiAnalysis(file_list=file_list, varnames=self.varnames)
            self.meta_stats.append(rkm.stats)

    def plot_data(self):

        for i, stats in enumerate(self.meta_stats):
            plt.errorbar(stats[:, 0], stats[:, 1], yerr=2*stats[:, 2],
                         fmt='.-', capsize=5, zorder=i)

        plt.title('Frequency Response Curves')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.legend(['GtACR - No Light', 'GtACR - With Light'])


if __name__ == '__main__':
    rkmeta = RkfMetaAnalysis(varnames=["ang_pos", "head_angle"])
    rkmeta.plot_data()
