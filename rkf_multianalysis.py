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

        self.tkroot = tk.Tk()
        filetypes = (("ROS bag files", "*.bag"), ("all files", "*.*"))
        file_list = tkf.askopenfilenames(initialdir=os.getcwd(), filetypes=filetypes)
        self.tkroot.destroy()

        pass

rkm = RkfMultiAnalysis()