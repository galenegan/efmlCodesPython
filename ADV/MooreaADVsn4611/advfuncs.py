#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 13:33:25 2018

@author: gegan
"""

import copy
from mylib import naninterp
import numpy as np
import pandas as pd

def xyz_enu(adv,heading,pitch,roll):
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
    
    for ii in range(1,np.max(list(adv['burst']))+1):
        Veltemp = np.stack((adv['burst'][ii]['velx'],adv['burst'][ii]['vely'],adv['burst'][ii]['velz']))
        Velnew = np.matmul(M,Veltemp)
        adv['burst'][ii]['velx'] = Velnew[0,:]
        adv['burst'][ii]['vely'] = Velnew[1,:]
        adv['burst'][ii]['velz'] = Velnew[2,:]
    
    adv['gen']['heading'] = heading
    adv['gen']['pitch'] = pitch
    adv['gen']['roll'] = roll
    
    return adv

def lpf(arr,fs,fc):
    
    arr = copy.deepcopy(arr)
    
    nanidx = np.isnan(arr)
    
    if np.sum(nanidx) > 0:
        arr = naninterp(arr)
    
    N = np.size(arr)
    
    #Windowing function (in this case, no window but could add later)
    H = 1
    
    F = (1./N)*np.fft.fftshift(np.fft.fft(arr*H))
    
    if N%2 == 1:
        s = np.arange((-(N-1)/2)*fs/N,((N-1)/2)*fs/N,fs/N)
    else:
        s = np.arange((-N/2)*fs/N,(N/2)*fs/N,fs/N)
    
    
    cutidx = np.where(np.abs(s)>fc)
    
    filt = np.ones(np.shape(s))
    filt[cutidx] = 0
    
    F_filt = F*filt
    arr_filt = (1./H)*np.real(np.fft.ifft(np.fft.ifftshift(F_filt*N)))
    
    arr_filt[0:4] = np.nan
    arr_filt[-5:] = np.nan
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

def get_roughness(adv,adv_filt,ref_height,vel_min,rho,doffp):
    
    #%% Constants and initializing dict
    k = 0.41 #von karman constant
    max_depth_frac = 0.6 #fraction of depth allowed for z0
    min_z0 = 1e-4 #minimum allowed z0 value
    
    roughness = dict()
    
    #%% Calculations
    for ii in range(1,np.max(list(adv['burst']))+1):
        
        roughness[ii] = dict()
        
        depth = adv['burst'][ii]['press']/(rho*9.81) + doffp
        
        #Reynolds stress
        up = adv['burst'][ii]['velmaj'] - adv_filt['burst'][ii]['velmaj']
        vp = adv['burst'][ii]['velmin'] - adv_filt['burst'][ii]['velmin']
        wp = adv['burst'][ii]['velz'] - adv_filt['burst'][ii]['velz']
        
        roughness[ii]['upwp'] = lpf(naninterp(up*wp), 
                 adv['gen']['fs'],adv['gen']['fs']/10)
        
        roughness[ii]['vpwp'] = lpf(naninterp(vp*wp),
                 adv['gen']['fs'],adv['gen']['fs']/10)
        
        #Friction velocity
        roughness[ii]['ustar'] = np.sqrt(np.abs(roughness[ii]['upwp']))
        roughness[ii]['vstar'] = np.sqrt(np.abs(roughness[ii]['vpwp']))
        
        #Roughness height
        roughness[ii]['z0maj'] = ref_height/np.exp(adv_filt['burst'][ii]['velmaj']*k/roughness[ii]['ustar'])
        roughness[ii]['z0min'] = ref_height/np.exp(adv_filt['burst'][ii]['velmin']*k/roughness[ii]['vstar'])
        
        #Removing values for calculating Cd
        badidx = np.abs(adv_filt['burst'][ii]['velmaj']) < vel_min
        roughness[ii]['z0maj'][badidx] = np.NaN
        
        badidx = np.abs(adv_filt['burst'][ii]['velmin']) < vel_min
        roughness[ii]['z0min'][badidx] = np.NaN
        
        roughness[ii]['z0maj'][roughness[ii]['z0maj']> max_depth_frac*depth/30] = np.NaN
        roughness[ii]['z0maj'][roughness[ii]['z0maj']< min_z0] = np.NaN
        
        roughness[ii]['z0min'][roughness[ii]['z0min']> max_depth_frac*depth/30] = np.NaN
        roughness[ii]['z0min'][roughness[ii]['z0min']< min_z0] = np.NaN
        
        #Computing Cd
        roughness[ii]['Cdmaj'] = k**2/(np.log(ref_height/roughness[ii]['z0maj'])**2)
        roughness[ii]['Cdmin'] = k**2/(np.log(ref_height/roughness[ii]['z0maj'])**2)
        
    return roughness
        
    
    
    
    