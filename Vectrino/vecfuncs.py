#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 16:41:08 2018

@author: gegan
"""

import copy
from mylib import naninterp
import numpy as np

def xyz_enu(vectrino,heading,pitch,roll):
    CH = np.cos(np.radians(heading))
    SH = np.sin(np.radians(heading))
    CP = np.cos(np.radians(pitch))
    SP = np.sin(np.radians(pitch))
    CR = np.cos(np.radians(roll))
    SR = np.sin(np.radians(roll))
    
    H = np.array(([CH,SH,0],[-SH,CH, 0],[0, 0, 1]))
    P = np.array(([1,0,0],[0, CP, -SP],[0,SP,CP]))
    R = np.array(([CR,0,SR],[0,1,0],[-SR,0,CR]))
    
    m1 = np.matmul(P,R)
    M = np.matmul(H,m1)
    
    for ii in range(np.max(list(vectrino['burst']))+1):
        for jj in range(np.shape(vectrino['burst'][ii]['u'])[0]):
            Veltemp1 = np.stack((vectrino['burst'][ii]['u'][jj,:],vectrino['burst'][ii]['v'][jj,:],
                                 vectrino['burst'][ii]['w1'][jj,:]))
            Veltemp2 = np.stack((vectrino['burst'][ii]['u'][jj,:],vectrino['burst'][ii]['v'][jj,:],
                                 vectrino['burst'][ii]['w2'][jj,:]))
            Velnew1 = np.matmul(M,Veltemp1)
            Velnew2 = np.matmul(M,Veltemp2)
            vectrino['burst'][ii]['u'][jj,:] = Velnew1[0,:]
            vectrino['burst'][ii]['v'][jj,:] = Velnew1[1,:]
            vectrino['burst'][ii]['w1'][jj,:] = Velnew1[2,:]
            vectrino['burst'][ii]['w2'][jj,:] = Velnew2[2,:]
    
    return vectrino

def lpf(arr,fs,fc):
    
    arr = copy.deepcopy(arr)
    
    for ii in range(np.size(arr,axis=1)):
        nanidx = np.isnan(arr[:,ii])
        if np.sum(nanidx) > 0:
            arr[:,ii] = naninterp(arr[:,ii])
      
    M,N = np.shape(arr)
    
    #Windowing function (in this case, no window but could add later)
    H = 1
    
    F = (1./N)*np.fft.fftshift(np.fft.fft2(arr*H))
    
    if N%2 == 1:
        s = np.arange((-(N-1)/2)*fs/N,((N)/2)*fs/N,fs/N)
    else:
        s = np.arange((-N/2)*fs/N,((N+1)/2 - 1)*fs/N,fs/N)
    
    S = np.ones((M,np.size(s)))
    for ii in range(M):
        S[ii,:] = S[ii,:]*s
    
    cutidx = np.where(np.abs(S)>fc)
    
    filt = np.ones((M,np.size(s)))
    filt[cutidx] = 0
    
    F_filt = F*filt
    arr_filt = (1./H)*np.real(np.fft.ifft2(np.fft.ifftshift(F_filt*N)))
    
    arr_filt[:,0:4] = np.nan
    arr_filt[:,-5:] = np.nan
    #arr_filt[nanidx] = np.nan #This puts original nans back in..probably not necessary 
    
    return arr_filt

def pa_theta(u,v):
    
    #Storing as complex variable w = u + iv
    w = u + 1j*v
    w = w[np.isfinite(w)]
    
    #Covariance matrix
    cv = np.cov(np.stack((np.real(w),np.imag(w))))
    
    #Theta is direction of maximum variance
    theta = (180./np.pi)*(0.5*np.arctan2(2.*cv[1,0],(cv[0,0]-cv[1,1])))
    
    return theta

def pa_rotation(u,v,theta):
    #Storing as complex variable w = u + iv
    w = u + 1j*v
    
    wr = w*np.exp(-1j*theta*np.pi/180)
    vel_maj = np.real(wr)
    vel_min = np.imag(wr)
    
    return vel_maj,vel_min