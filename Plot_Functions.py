# -*- coding: utf-8 -*-
"""
Plot_Funcs
Author: Grant Stephens
"""
import numpy as np  # NumPy (multidimensional arrays, linear algebra, ...)
import scipy as sp  # SciPy (signal and image processing library)
import matplotlib as mpl         # Matplotlib (2D/3D plotting library)
import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
from pylab import *              # Matplotlib's pylab interface
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def surf(z,x=0,y=0):
    ydim, xdim = np.shape(z)    
    if np.size(x)==1:
        x=linspace(0,1,xdim)
    if np.size(y)==1:
        y=linspace(0,1,ydim)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=True)  
    ax.set_zlim3d(np.min(z), np.max(z))
    show()