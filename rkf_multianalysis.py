import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.signal as sps
import numpy as np
import os
import Tkinter as tk
import tkFileDialog as tkf
import scipy.signal as sps
from py_pll import PyPLL

from rkf_analysis import RkfAnalysis, Variable


class RkfMultiAnalysis(object):

    def __init__(self):

        tkroot = tk.Tk()
        filetypes = (("ROS bag files", "*.bag"), ("all files", "*.*"))
        self.file_list = tkf.askopenfilenames(initialdir=os.getcwd(), filetypes=filetypes)
        tkroot.destroy()

        self.cgain = np.empty((0, 2))
        self._calc_compound_gain()

    def _calc_compound_gain(self):

        for file in self.file_list:

            rka = RkfAnalysis(file)
            self.cgain = np.concatenate((self.cgain, np.array(rka.calc_response(rka.ang_pos, rka.wing_diff))), axis=0)

        self.cgain = self.cgain[np.argsort(self.cgain[:, 0])]

rkm = RkfMultiAnalysis()