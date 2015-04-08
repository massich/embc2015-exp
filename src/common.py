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

import seaborn as sns

class myScatterGrid(sns.PairGrid):
    """ This class adapt the Seaborn's PariGrid class in order to use makrersize
    in a scatter plot. """

    def map_lower(self, func, **kwargs):
        """Plot with a bivariate function on the lower diagonal subplots.
        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes.
        """
        kw_color = kwargs.pop("color", None)
        kw_s = kwargs.pop("s", None)
        for i, j in zip(*np.tril_indices_from(self.axes, -1)):
            hue_grouped = self.data.groupby(self.hue_vals)
            for k, (label_k, data_k) in enumerate(hue_grouped):

                ax = self.axes[i, j]
                plt.sca(ax)

                x_var = self.x_vars[j]
                y_var = self.y_vars[i]

                # Insert the other hue aesthetics if appropriate
                for kw, val_list in self.hue_kws.items():
                    kwargs[kw] = val_list[k]

                color = self.palette[k] if kw_color is None else kw_color
                if kw_s is None:
                    func(data_k[x_var], data_k[y_var], label=label_k,
                         color=color, **kwargs)
                else:
                    func(data_k[x_var], data_k[y_var], label=label_k,
                         color=color, s=kw_s[k], **kwargs)

            self._clean_axis(ax)
            self._update_legend_data(ax)

        if kw_color is not None:
            kwargs["color"] = kw_color
        self._add_axis_labels()


    def map_upper(self, func, **kwargs):
        """Plot with a bivariate function on the upper diagonal subplots.
        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes.
        """
        kw_color = kwargs.pop("color", None)
        kw_s = kwargs.pop("s", None)
        for i, j in zip(*np.triu_indices_from(self.axes, 1)):

            hue_grouped = self.data.groupby(self.hue_vals)
            for k, (label_k, data_k) in enumerate(hue_grouped):

                ax = self.axes[i, j]
                plt.sca(ax)

                x_var = self.x_vars[j]
                y_var = self.y_vars[i]

                # Insert the other hue aesthetics if appropriate
                for kw, val_list in self.hue_kws.items():
                    kwargs[kw] = val_list[k]

                color = self.palette[k] if kw_color is None else kw_color
                if kw_s is None :
                    func(data_k[x_var], data_k[y_var], label=label_k,
                         color=color, **kwargs)
                else:
                    func(data_k[x_var], data_k[y_var], label=label_k,
                         color=color, s=kw_s[k], **kwargs)

            self._clean_axis(ax)
            self._update_legend_data(ax)

        if kw_color is not None:
            kwargs["color"] = kw_color
        self._add_axis_labels()
