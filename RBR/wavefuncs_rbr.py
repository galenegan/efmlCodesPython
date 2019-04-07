#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 11:47:07 2018

@author: gegan
"""

"""Functions for calculating wave spectra and other wave statistics. Based
mostly on MATLAB code written by Falk Feddersen and Justin Rogers"""
import os
parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

import sys
sys.path.append(parentDir + '/General/')

import copy
from mylib import naninterp
import numpy as np
import scipy.signal as sig


def get_wavenumber(omega,h):
    
    """Returns wavenumber from the surface gravity wave dispersion relation
    using Newton's method"""
    
    g = 9.81
    k = omega/np.sqrt(g*h)
    
    f = g*k*np.tanh(k*h) - omega**2
    
    while np.max(np.abs(f)) > 1e-10:
        dfdk = g*k*h*((1/np.cosh(k*h))**2) + g*np.tanh(k*h)
        k = k - f/dfdk
        f = g*k*np.tanh(k*h) - omega**2
        
    return k

def get_cg(k,depth):
    """Returns the wave group velocity given the wavenumber and mean depth"""
    g=9.81
    Cp = np.sqrt((g/k)*np.tanh(k*depth))
    Cg = 0.5*Cp*(1+(k*depth)*(1-(np.tanh(k*depth))**2)/np.tanh(k*depth))
    
    return Cg

def calculate_fft2(x,nfft):
    
    X = copy.deepcopy(x)
    
    if X.ndim==1:
        n = np.size(X)
    elif X.ndim==2:
        n,m = np.shape(X)
        if m>n:
            X = X.T
            n,m = np.shape(X)
    
    num = int(np.floor(4*n/nfft) - 3)
    
    X = X - np.mean(X)
    
    jj = np.arange(0,nfft)
    
    WIN = np.hanning(np.size(jj))
    
    A = np.zeros((num,nfft),dtype = np.complex128)

    varXwindtot = 0    
    
    for ii in range(num):
        istart = int(np.ceil(ii*nfft/4))
        istop = int(np.floor(istart + nfft))  
        Xwind = X[istart:istop].squeeze()
        lenX = len(Xwind)
        Xwind = Xwind - np.mean(Xwind) #de-mean
        varXwind = np.dot(Xwind,Xwind)/lenX
        Xwind = sig.detrend(Xwind)
        
        varXwindtot = varXwindtot + varXwind
        Xwind = Xwind*WIN
        tmp = np.dot(Xwind,Xwind)/lenX
        
        if tmp == 0:
            Xwind = Xwind*0
        else:
            Xwind = Xwind*np.sqrt(varXwind/tmp)
        
        A[ii,:] = np.fft.fft(Xwind.T)/np.sqrt(nfft)
        
    
    return A
    
    

def wave_stats_spectra(p,depth,nfft,doffp,fs,fc,rho):
    
    """Calculates a number of common wave statistics.
    Inputs:
       p: vector of pressure readings in decibars
       d: vector of depth readings in meters
       nfft: Window size when calculating the power spectra in sig.welch
       doffp: pressure sensor height above bed in meters
       fs: sampling frequency in Hz
       fc: cutoff frequency in Hz, to avoid noise at higher frequencies
       rho: Water density (kg/m^3)
       
   Returns:
       Wavestats, a dictionary containing calculated wave statistics
    """

    P = copy.deepcopy(p)
    
    g = 9.81
    
    
    #Interpolating out nans
    P = naninterp(P)
    
    dbar = np.nanmean(depth) #Average water depth
    
    #Making sure average depth is positive
    if dbar<0:
        raise ValueError
    
    
    df = fs/(nfft-1) #frequency resolution

    
    fm, Spp = sig.welch(P,fs = fs,window = 'hamming',nperseg = nfft,detrend = 'linear')
    
    #Depth correction and spectral weighted averages
    f_ig_low = 1./250
    f_ig_high = 1./33.33
    f_swell_low = 1./25
    f_swell_high = 0.45
    
    i_ig = np.where((fm>f_ig_low) & (fm<f_ig_high))
    i_swell = np.where((fm>f_swell_low) & (fm<f_swell_high))
    i_all = np.where((fm>f_ig_low) & (fm<fc))
    
    omega = 2*np.pi*fm
    
    k = get_wavenumber(omega,dbar)
    
    correction = np.zeros((np.size(Spp),))
    
    ii = np.where(fm<=fc)
    
    #Correcting for pressure attenuation
    correction[ii] = 1e4*np.cosh(k[ii]*dbar)/(rho*g*np.cosh(k[ii]*doffp))
    
    SSE = Spp*(correction**2)
    
    SSEt = SSE[ii]

    fmt = fm[ii]

    #Energy flux
    Cg = get_cg(k,dbar)    
    
    
    #Significant wave height
    Hsigt = 4*np.sqrt(SSE[ii]*df)
    Hsig_ss = 4*np.sqrt(np.nansum(SSE[i_swell]*df))
    Hsig_ig = 4*np.sqrt(np.nansum(SSE[i_ig]*df))
    Hsig_all = 4*np.sqrt(np.nansum(SSE[i_all]*df))
    
    Hrmst = np.sqrt(8*SSE[ii]*df)
    Hrms_ss = np.sqrt(8*np.nansum(SSE[i_swell]*df))
    Hrms_ig = np.sqrt(8*np.nansum(SSE[i_ig]*df))
    Hrms_all = np.sqrt(8*np.nansum(SSE[i_all]*df))

    
    #Centroid frequency
    fcentroid_swell = np.nansum(fm[i_swell]*SSE[i_swell])/np.nansum(SSE[i_swell])
    Tm_ss = 1./fcentroid_swell
    
    #peak frequency
    indx = np.argmax(SSE[i_swell])
    temp = fmt[i_swell]
    Tp_ss = 1./temp[indx]
    if np.size(Tp_ss) == 0:
        Tp_ss = np.nan

    
    #Centroid frequency
    fcentroid_ig = np.nansum(fm[i_ig]*SSE[i_ig])/np.nansum(SSE[i_ig])
    Tm_ig = 1./fcentroid_ig
    
    #peak frequency
    indx = np.argmax(SSE[i_ig])
    temp = fmt[i_ig]
    Tp_ig = 1./temp[indx]
    if np.size(Tp_ig) == 0:
        Tp_ig = np.nan
        
    #Centroid frequency
    fcentroid_all = np.nansum(fm[i_all]*SSE[i_all])/np.nansum(SSE[i_all])
    Tm_all = 1./fcentroid_all
    
    #All peak
    indx = np.argmax(SSE[i_all])
    temp = fmt[i_all]
    Tp_all = 1./temp[indx]
    if np.size(Tp_all) == 0:
        Tp_all = np.nan

    #output
    Wavestats = dict()
    Wavestats['SSEt'] = SSEt
    Wavestats['fmt'] = fmt
    Wavestats['Tt'] = 1/fmt
    Wavestats['Hsigt'] = Hsigt
    Wavestats['Hsig_ss'] = Hsig_ss
    Wavestats['Hsig_ig'] = Hsig_ig
    Wavestats['Hsig_all'] = Hsig_all
    Wavestats['Hrmst'] = Hrmst
    Wavestats['Hrms_ss'] = Hrms_ss
    Wavestats['Hrms_ig'] = Hrms_ig
    Wavestats['Hrms_all'] = Hrms_all
    Wavestats['Tm_ss'] = Tm_ss
    Wavestats['Tm_ig'] = Tm_ig
    Wavestats['Tm_all'] = Tm_all
    Wavestats['Tp_ss'] = Tp_ss
    Wavestats['Tp_ig'] = Tp_ig
    Wavestats['Tp_all'] = Tp_all
    Wavestats['depth'] = depth
    Wavestats['df'] = df
    Wavestats['Cg'] = Cg

    return Wavestats
