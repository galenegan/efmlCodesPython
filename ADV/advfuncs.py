#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 13:33:25 2018

@author: gegan
"""
import copy
from mylib import naninterp
import numpy as np
import scipy.signal as sig
import datetime

def xyz_enu(adv,headingin,pitch,roll,switch_times):
    
    
    for ii in adv:
        if adv[ii]['burststart']<switch_times[0]:
            heading = headingin[0]
        elif (adv[ii]['burststart']>switch_times[0]) & (adv[ii]['burststart']<switch_times[1]):
            heading = headingin[1]
        else:
            heading = headingin[2]
       
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
    
    
        Veltemp = np.stack((adv[ii]['velx'],adv[ii]['vely'],adv[ii]['velz']))
        Velnew = np.matmul(M,Veltemp)
        adv[ii]['velx'] = Velnew[0,:]
        adv[ii]['vely'] = Velnew[1,:]
        adv[ii]['velz'] = Velnew[2,:]
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
        s = np.arange((-(N-1)/2)*fs/N,((N)/2)*fs/N,fs/N)
    else:
        s = np.arange((-N/2)*fs/N,((N+1)/2 - 1)*fs/N,fs/N)
    
    
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

def get_dissipation(adv,fs):
    
    def calcA13(sigV,theta):
       x = np.arange(-20,20,1e-2)
       out = np.empty_like(sigV.flatten())
       for i, (b,t) in enumerate(zip(sigV.flatten(),theta.flatten())):
           out[i] = np.trapz(np.cbrt(x ** 2 - 2 / b * np.cos(t) * x + 
              b ** (-2)) *np.exp(-0.5 * x ** 2), x)
    
       return out.reshape(sigV.shape)*(2*np.pi)**(-0.5) * sigV**(2/3)
       
    def up_angle(u,v):
        Uh = naninterp(u) + 1j*naninterp(v)
        dt = sig.detrend(Uh)
        fx = dt.imag <= 0
        dt[fx] = dt[fx]*np.exp(1j * np.pi)
        return np.angle(np.mean(dt,-1,dtype = np.complex128))
    
    def U_angle(u,v):
        n = np.nanmean(v)
        e = np.nanmean(u)
        return np.arctan2(n,e)
    
    omega_range = [2*np.pi*1.2,2*np.pi*2]
    #omega_range = [2*np.pi*4,2*np.pi*9]
    #omega_range = [2*np.pi*2,2*np.pi*3]
    #omega_range = [2*np.pi*1.2,2*np.pi*2]

    u = adv['velmaj']
    v = adv['velmin']
    w = adv['velz']
    
    if np.sum(np.isnan(u)) < len(u)/2:
        
        u = naninterp(u)
        v = naninterp(v)
        w = naninterp(w)
        
        V = np.sqrt(np.nanmean(u**2 + v**2))
        sigma = np.std(np.sqrt(u**2+v**2))
        
        thetaup = up_angle(u,v)
        thetaU = U_angle(u,v)
        theta = thetaU - thetaup
        
        alpha = 1.5
        intgrl = calcA13(sigma/V,theta)
        
        fu,Pu = sig.welch(u,fs = fs, window = 'hamming', nperseg = len(u)//50,
                          detrend = 'linear')
        fv,Pv = sig.welch(v,fs = fs, window = 'hamming', nperseg = len(v)//50,
                          detrend = 'linear')
        fw,Pw = sig.welch(w,fs = fs, window = 'hamming', nperseg = len(w)//50,
                          detrend = 'linear')
        
        noiserange = (fu>=3) & (fu<=4)
        #noiserange = (fu >= 3) & (fu <= 6)
        noiselevel = np.nanmean(Pu[noiserange] + Pv[noiserange])
        
        omega = 2*np.pi*fu
        inds = (omega > omega_range[0]) & (omega < omega_range[1])
        omega = omega[inds]
        Pu = Pu[inds]
        Pv = Pv[inds]
        Pw = Pw[inds]
        
        uv = (np.mean((Pu + Pv - noiselevel)*(omega)**(5/3))/
              (21/55*alpha*intgrl))**(3/2)/V

        #Adding w component
        uv = (np.mean((Pw)*(omega)**(5/3))/
              (12/55*alpha*intgrl))**(3/2)/V
        
        # Averaging
        uv *= 0.5
        
        eps = uv
    else:
        eps = np.NaN
        
    return eps


    
    
    