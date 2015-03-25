import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('glemaitre', 'se04g0bmi2')

import mpld3

import numpy as np

from scipy.stats import norm, rayleigh, rice
from scipy.optimize import curve_fit


class Patient(object):
    """ A class with a single method __init__ should never be a class
    (or coded as a class)
    """
    def __init__(self, data):
        self.data = data
        self.pdf, self.bin_edges = np.histogram(
            self.data,
            bins=(np.max(self.data) - np.min(self.data)),
            density=True
            )
        self.max_int = np.max(self.data)
        self.min_int = np.min(self.data)
        self.rayleigh_params = rayleigh.fit(self.data)
        self.gaussian_params = norm.fit(self.data)
        # self.rice_params = rice.fit(self.data)


