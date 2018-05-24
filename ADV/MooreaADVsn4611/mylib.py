#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:50:07 2018

@author: gegan
"""

import copy
import numpy as np
import scipy.interpolate


def naninterp(x):
    
    if np.max(np.isnan(x)) == 1:
        f = scipy.interpolate.interp1d(np.reshape(np.array(np.where(~np.isnan(x))),(np.size(x[~np.isnan(x)]),)),x[~np.isnan(x)],
         kind = 'linear',bounds_error =False)
        xnew = np.where(np.isnan(x))
        x[np.isnan(x)]=f(xnew)
    
    if np.max(np.isnan(x)) == 1:
        f = scipy.interpolate.interp1d(np.reshape(np.array(np.where(~np.isnan(x))),(np.size(x[~np.isnan(x)]),)),x[~np.isnan(x)],
         kind = 'nearest',fill_value = 'extrapolate')
        xnew = np.where(np.isnan(x))
        x[np.isnan(x)]=f(xnew)
    
    return x