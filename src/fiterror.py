import matplotlib.pyplot as plt
%matplotlib gtk
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D

import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('glemaitre', 'se04g0bmi2')

import mpld3

import numpy as np

from scipy.stats import norm, rayleigh, rice
from scipy.optimize import curve_fit

import pandas as pd
import seaborn as sns
import random

data = pd.read_csv('../scratch/patient00_data.csv',
                   header=[0,1],index_col=[0])
fit = pd.read_csv('../scratch/patient00_myGLF_fitting.csv')
estim = pd.read_csv('../scratch/patient00_myGLF_genval.csv')

fit['label'] = data.loc[:,('label','gt')].reset_index(drop=True)

#g = sns.PairGrid(fit[pd.notnull(fit.A)],['A','B', 'K','Q','v'],hue='label')
#g.map_diag(plt.hist)
#g.map_offdiag(plt.scatter, alpha=.5)
#
#sns.violinplot(fit[pd.notnull(fit.A)].A, groupby=fit[pd.notnull(fit.A)].label)

def subsample(d,n_samples=1000):
    # Get less data
    d = d[pd.notnull(d.A)].reset_index(drop=True)
    d = d[pd.notnull(d.label)].reset_index(drop=True)
    d.set_index('sampleId', inplace=True) 
    rows = np.array([])
    for label in d.label.unique():
        rows = np.append(rows,
                random.sample(d[d.label==label].index, n_samples))

    d = d.ix[rows]
    return d

d = subsample(fit, 2000)
#    with sns.color_palette("Set2"):
#        for label in d.label.unique():
#            sns.distplot(d[d.label==label].A)
%matplotlib gtk
g = sns.PairGrid(d,vars=['A','B', 'K','Q','v'],hue='label', palette="Set2")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter, alpha=.5)
g.map_lower(sns.kdeplot)  


