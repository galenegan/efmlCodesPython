#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:50:07 2018

@author: gegan
"""

import datetime
import numpy as np
import scipy.interpolate



def naninterp(x):
    
    """Linear interpolation over NaN values in an array"""
    
    if ~np.all(np.isnan(x)):
    
        if np.sum(~np.isnan(x)) >= 2:
            f = scipy.interpolate.interp1d(np.reshape(np.array(np.where(~np.isnan(x))),(np.size(x[~np.isnan(x)]),)),x[~np.isnan(x)],
             kind = 'linear',bounds_error =False)
            xnew = np.where(np.isnan(x))
            x[np.isnan(x)]=f(xnew).squeeze()
        
        if np.sum(~np.isnan(x)) >= 2:
            f = scipy.interpolate.interp1d(np.reshape(np.array(np.where(~np.isnan(x))),(np.size(x[~np.isnan(x)]),)),x[~np.isnan(x)],
             kind = 'nearest',fill_value = 'extrapolate')
            xnew = np.where(np.isnan(x))
            x[np.isnan(x)]=f(xnew).squeeze()
    
    return x

def m2ptime(mtime):
    
    """Convert matlab datenums to datetime.datetime"""
    
    ptime = np.zeros(np.shape(mtime),dtype = datetime.datetime)
    
    for ii in range(len(mtime)):
        ptime[ii] = datetime.datetime.fromordinal(int(mtime[ii])) + datetime.timedelta(days=mtime[ii]%1) - datetime.timedelta(days = 366)
    
    return ptime

def p2mtime(ptime):
    
    """Convert datetime.datetime to matlab datenum"""
    
    mtime = np.zeros(np.shape(ptime),dtype = float)
        
    for ii in range(len(ptime)):
        mtime[ii] = (int(datetime.datetime.toordinal(ptime[ii] + datetime.timedelta(days = 366))) + (ptime[ii].hour)/24. + (ptime[ii].minute)/1440.
             + (ptime[ii].second)/86400. + (ptime[ii].microsecond)/(86400.*1e6))
    
    return mtime

def get_r2(fit,ydata):
    
    residuals = ydata-fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.nanmean(ydata))**2)
    
    r2 = 1 - (ss_res/ss_tot)
    return r2
