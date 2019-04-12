# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:13:37 2017

@author: gegan
"""
#Reads in data from the Vectrino II profiler

#Packages
import os
parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

import sys
sys.path.append(parentDir + '/General/')

import copy
import datetime
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import scipy.ndimage 
import vecfuncs
import pandas as pd
import mylib


#Constants
x_heading = 185.
vertical_orientation = 'down'
corr_min = 10.
snr_min = 20
filtstyle = 'butter'
wavedecomp = True
fs = 64

#%%Loading in data and assigning to variables
filepath = ''
files = glob.glob(filepath + '*.mat')
files.sort()
Nfiles = np.size(files)

burstnums = []

for file in files:
    bnum = file.split('-')[1].split('.data')[0]
    burstnums = np.concatenate((burstnums,[int(bnum)]))
burstnums = np.unique(burstnums,return_index = True)
burstnum = burstnums[0].astype(int)
burstind = burstnums[1]

#Assuming that each .mat file is split into two parts
for ii in range(len(burstind)-1):
    vectrino = dict()
    
    if burstind[ii+1] == burstind[ii] + 2:
        
        idx = burstind[ii]
        matdata1 = sio.loadmat(files[idx],matlab_compatible=True)
        Data1 = matdata1[list(matdata1.keys())[3]]
        Config1 = matdata1[list(matdata1.keys())[4]]
        
        matdata2 = sio.loadmat(files[idx+1],matlab_compatible=True)
        Data2 = matdata2[list(matdata2.keys())[3]]
        Config2 = matdata2[list(matdata2.keys())[4]]
 
        #Time 
        tstart = datetime.datetime.fromtimestamp(int(files[idx][56:72])/1e6)
        dt = np.concatenate((Data1['Profiles_HostTime'].item().squeeze().T,
                             Data2['Profiles_HostTime'].item().squeeze().T))
        t = tstart+datetime.timedelta(seconds=1)*dt
        t = t + datetime.timedelta(hours = 7) - datetime.timedelta(seconds = dt[-1])
        vectrino['time'] = t
        
        #Bottom distance and vertical range
        bottom_dist = np.concatenate((Data1['BottomCheck_BottomDistance'].item().squeeze(),
                                      Data2['BottomCheck_BottomDistance'].item().squeeze()))
    
        if np.all(bottom_dist == 0):
            vectrino['bot'] = 0.06
        else:
            vectrino['bot'] = np.nanmean(bottom_dist[bottom_dist!=0])
    
        vrange = Data1['Profiles_Range'].item().T.squeeze()
        vectrino['z']   = -vrange + vectrino['bot']
    
        #The rest of the data
        vectrino['u'] = np.concatenate((Data1['Profiles_VelX'].item().squeeze(),Data2['Profiles_VelX'].item().squeeze())).T
        vectrino['v'] = np.concatenate((Data1['Profiles_VelY'].item().squeeze(),Data2['Profiles_VelY'].item().squeeze())).T
        vectrino['w1'] = np.concatenate((Data1['Profiles_VelZ1'].item().squeeze(),Data2['Profiles_VelZ1'].item().squeeze())).T
        vectrino['w2'] = np.concatenate((Data1['Profiles_VelZ2'].item().squeeze(),Data2['Profiles_VelZ2'].item().squeeze())).T
        
        vectrino['corr1'] = np.concatenate((Data1['Profiles_CorBeam1'].item().squeeze(),Data2['Profiles_CorBeam1'].item().squeeze())).T
        vectrino['corr2'] = np.concatenate((Data1['Profiles_CorBeam2'].item().squeeze(),Data2['Profiles_CorBeam2'].item().squeeze())).T
        vectrino['corr3'] = np.concatenate((Data1['Profiles_CorBeam3'].item().squeeze(),Data2['Profiles_CorBeam3'].item().squeeze())).T
        vectrino['corr4'] = np.concatenate((Data1['Profiles_CorBeam4'].item().squeeze(),Data2['Profiles_CorBeam4'].item().squeeze())).T
        
        vectrino['amp1'] = np.concatenate((Data1['Profiles_AmpBeam1'].item().squeeze(),Data2['Profiles_AmpBeam1'].item().squeeze())).T
        vectrino['amp2'] = np.concatenate((Data1['Profiles_AmpBeam2'].item().squeeze(),Data2['Profiles_AmpBeam2'].item().squeeze())).T
        vectrino['amp3'] = np.concatenate((Data1['Profiles_AmpBeam3'].item().squeeze(),Data2['Profiles_AmpBeam3'].item().squeeze())).T
        vectrino['amp4'] = np.concatenate((Data1['Profiles_AmpBeam4'].item().squeeze(),Data2['Profiles_AmpBeam4'].item().squeeze())).T
    
        vectrino['snr1'] = np.concatenate((Data1['Profiles_SNRBeam1'].item().squeeze(),Data2['Profiles_SNRBeam1'].item().squeeze())).T
        vectrino['snr2'] = np.concatenate((Data1['Profiles_SNRBeam2'].item().squeeze(),Data2['Profiles_SNRBeam2'].item().squeeze())).T
        vectrino['snr3'] = np.concatenate((Data1['Profiles_SNRBeam3'].item().squeeze(),Data2['Profiles_SNRBeam3'].item().squeeze())).T
        vectrino['snr4'] = np.concatenate((Data1['Profiles_SNRBeam4'].item().squeeze(),Data2['Profiles_SNRBeam4'].item().squeeze())).T
    else:
        idx = burstind[ii]
        matdata1 = sio.loadmat(files[idx],matlab_compatible=True)
        Data1 = matdata1[list(matdata1.keys())[3]]
        Config1 = matdata1[list(matdata1.keys())[4]]
        
        #Time 
        tstart = datetime.datetime.fromtimestamp(int(files[idx][56:72])/1e6)
        dt = Data1['Profiles_HostTime'].item().squeeze().T
        t = tstart+datetime.timedelta(seconds=1)*dt
        t = t + datetime.timedelta(hours = 7) - datetime.timedelta(seconds = dt[-1])
        vectrino['time'] = t
        
        #Bottom distance and vertical range
        bottom_dist = Data1['BottomCheck_BottomDistance'].item().squeeze()
    
        if np.all(bottom_dist == 0):
            vectrino['bot'] = 0.06
        else:
            vectrino['bot'] = np.nanmean(bottom_dist[bottom_dist!=0])
    
        vrange = Data1['Profiles_Range'].item().T.squeeze()
        vectrino['z']   = -vrange + vectrino['bot']
    
        #The rest of the data
        vectrino['u'] = Data1['Profiles_VelX'].item().squeeze().T
        vectrino['v'] = Data1['Profiles_VelY'].item().squeeze().T
        vectrino['w1'] = Data1['Profiles_VelZ1'].item().squeeze().T
        vectrino['w2'] = Data1['Profiles_VelZ2'].item().squeeze().T
        
        vectrino['corr1'] = Data1['Profiles_CorBeam1'].item().squeeze().T
        vectrino['corr2'] = Data1['Profiles_CorBeam2'].item().squeeze().T
        vectrino['corr3'] = Data1['Profiles_CorBeam3'].item().squeeze().T
        vectrino['corr4'] = Data1['Profiles_CorBeam4'].item().squeeze().T
        
        vectrino['amp1'] = Data1['Profiles_AmpBeam1'].item().squeeze().T
        vectrino['amp2'] = Data1['Profiles_AmpBeam2'].item().squeeze().T
        vectrino['amp3'] = Data1['Profiles_AmpBeam3'].item().squeeze().T
        vectrino['amp4'] = Data1['Profiles_AmpBeam4'].item().squeeze().T
    
        vectrino['snr1'] = Data1['Profiles_SNRBeam1'].item().squeeze().T
        vectrino['snr2'] = Data1['Profiles_SNRBeam2'].item().squeeze().T
        vectrino['snr3'] = Data1['Profiles_SNRBeam3'].item().squeeze().T
        vectrino['snr4'] = Data1['Profiles_SNRBeam4'].item().squeeze().T
    # Rotating XYZ to earth coordinates
    if vertical_orientation == 'up':
        roll = 180
        pitch = 0
        heading = x_heading + 90
    elif vertical_orientation == 'down':
        roll = 0
        pitch = 0
        heading = x_heading-90
    
    vectrino = vecfuncs.xyz_enu(vectrino,heading,pitch,roll)
    
    #NaN below the bed
    trimbins = (vectrino['z'] < 0)
    
    
    vectrino['u'][trimbins,:] = np.NaN
    vectrino['v'][trimbins,:] = np.NaN
    vectrino['w1'][trimbins,:] = np.NaN
    vectrino['w2'][trimbins,:] = np.NaN
    
    vectrino['corr1'][trimbins,:] = np.NaN
    vectrino['corr2'][trimbins,:] = np.NaN
    vectrino['corr3'][trimbins,:] = np.NaN
    vectrino['corr4'][trimbins,:] = np.NaN
    
    vectrino['amp1'][trimbins,:] = np.NaN
    vectrino['amp2'][trimbins,:] = np.NaN
    vectrino['amp3'][trimbins,:] = np.NaN
    vectrino['amp4'][trimbins,:] = np.NaN
    
    vectrino['snr1'][trimbins,:] = np.NaN
    vectrino['snr2'][trimbins,:] = np.NaN
    vectrino['snr3'][trimbins,:] = np.NaN
    vectrino['snr4'][trimbins,:] = np.NaN
      
    #Removing values with correlations below corr_min and snr below snr_min
    badidx1 = (vectrino['corr1']<corr_min) | (vectrino['snr1']<snr_min)
    badidx2 = (vectrino['corr2']<corr_min) | (vectrino['snr2']<snr_min)
    badidx3 = (vectrino['corr3']<corr_min) | (vectrino['snr3']<snr_min)
    badidx4 = (vectrino['corr4']<corr_min) | (vectrino['snr4']<snr_min)
    
    
    vectrino['u'][badidx1] = np.NaN
    vectrino['v'][badidx2] = np.NaN
    vectrino['w1'][badidx3] = np.NaN
    vectrino['w2'][badidx4] = np.NaN      
    

    #Calculating principal axis rotation
    theta =  vecfuncs.pa_theta(np.nanmean(vectrino['u'],axis=0),
            np.nanmean(vectrino['v'],axis=0))
    vectrino['theta'] = theta
    
    vectrino['velmaj'], vectrino['velmin'] = vecfuncs.pa_rotation(
            vectrino['u'],vectrino['v'],vectrino['theta'])
    
    #Low-pass filtering the velocity and pressure data and placing in separate 
    #structure 
    
    fc = 1./30 #Cutoff frequency for fourier and butterworth filters
    filt_kernel =  (3,9) #kernel for median filter

#    # Method 1--fourier filter
#    if filtstyle == 'fourier':
#
#        vectrino_filt[ii]['u'] = vecfuncs.lpf(vectrino['u'],
#                fs,fc) 
#        
#        vectrino_filt[ii]['v'] = vecfuncs.lpf(vectrino['v'],
#                fs,fc) 
#        
#        vectrino_filt[ii]['w1'] = vecfuncs.lpf(vectrino['w1'],
#                fs,fc) 
#        
#        vectrino_filt[ii]['w2'] = vecfuncs.lpf(vectrino['w2'],
#                fs,fc) 
#
#    #Method 2: median filter 
#    elif filtstyle == 'median':
#    
#        #filt_kernel = 9
#        vectrino_filt[ii]['u'] = scipy.ndimage.filters.median_filter(vectrino['u'],
#                     size = filt_kernel,mode = 'nearest')
#        vectrino_filt[ii]['v'] = scipy.ndimage.filters.median_filter(vectrino['v'],
#                     size = filt_kernel,mode = 'nearest')
#        vectrino_filt[ii]['w1'] = scipy.ndimage.filters.median_filter(vectrino['w1'],
#                     size = filt_kernel,mode = 'nearest')
#        vectrino_filt[ii]['w2'] = scipy.ndimage.filters.median_filter(vectrino['w2'],
#                     size = filt_kernel,mode = 'nearest')
#        vectrino_filt[ii]['velmaj'] = scipy.ndimage.filters.median_filter(vectrino['velmaj'],
#                     size = filt_kernel,mode = 'nearest')
#        vectrino_filt[ii]['velmin'] = scipy.ndimage.filters.median_filter(vectrino['velmin'],
#                     size = filt_kernel,mode = 'nearest')
#        vectrino_filt[ii]['SSC'] = scipy.ndimage.filters.median_filter(vectrino['SSC'],
#                     size = filt_kernel[1],mode = 'nearest')
#        
#    elif filtstyle == 'butter':
#        Wn = fc/(fs/2)
#        b,a = sig.butter(2,Wn,btype = 'low')
#        vectrino_filt[ii]['u'] = np.empty(np.shape(vectrino['u']))
#        vectrino_filt[ii]['v'] = np.empty(np.shape(vectrino['v']))
#        vectrino_filt[ii]['w1'] = np.empty(np.shape(vectrino['w1']))
#        vectrino_filt[ii]['w2'] = np.empty(np.shape(vectrino['w2']))
#        vectrino_filt[ii]['velmaj'] = np.empty(np.shape(vectrino['velmaj']))
#        vectrino_filt[ii]['velmin'] = np.empty(np.shape(vectrino['velmin']))
#        
#        for jj in range(np.size(vrange)):
#            u = mylib.naninterp(vectrino['u'][jj,:])
#            v = mylib.naninterp(vectrino['v'][jj,:])
#            w1 = mylib.naninterp(vectrino['w1'][jj,:])
#            w2 = mylib.naninterp(vectrino['w2'][jj,:])
#            velmaj = mylib.naninterp(vectrino['velmaj'][jj,:])
#            velmin = mylib.naninterp(vectrino['velmin'][jj,:])
#            vectrino_filt[ii]['u'][jj,:] = sig.filtfilt(b,a,u)
#            vectrino_filt[ii]['v'][jj,:] = sig.filtfilt(b,a,v)
#            vectrino_filt[ii]['w1'][jj,:] = sig.filtfilt(b,a,w1)
#            vectrino_filt[ii]['w2'][jj,:] = sig.filtfilt(b,a,w2)
#            vectrino_filt[ii]['velmaj'][jj,:] = sig.filtfilt(b,a,velmaj)
#            vectrino_filt[ii]['velmin'][jj,:] = sig.filtfilt(b,a,velmin)

    
    # Calculating Reynolds stress tensor

#    #Can choose either phase method or benilov method. Benilov requires pressure data. 
#    if wavedecomp:
#        method = 'phase'
#        waveturb[ii] = vecfuncs.get_turb_waves(vectrino,fs,method)
#    
#    #Can change filtstyle to nofilt to calculate reynolds stress based solely on covariance (i.e. one data point per burst per cell)
#   
#    #Calculating other turbulence statistics, including non wave-decomposed covariances
#    # and log law fitting
#    filt_style_cov = 'nofilt'
#    turb[ii] = vecfuncs.get_turb(vectrino,vectrino_filt[ii],fs,fc,filt_kernel,filt_style_cov)
#        
##    #Correcting with Thomas et al 2017. Only do this if reynolds stress calculation
##    #gives one value per burst (i.e. wave turbulence decomposition, or nofilt covariance method)
#    RScorr[ii] = vecfuncs.noise_correction(turb[ii])
    
    #Saving the data...there's a bug in pickle that doesn't allow saving the entire thing 
        #because it's too big. So saving each burst individually
#    
    np.save('/Volumes/TOURO S/Deployment1/Vectrino/npyfix/vectrino_'+str(ii),vectrino)
#    np.save('/Volumes/TOURO S/Deployment1/Vectrino/npyfix/vectrinofilt_'+str(ii),vectrino_filt[ii])
#    np.save('/Volumes/TOURO S/Deployment1/Vectrino/npyfix/waveturb_'+str(ii),waveturb[ii])
#    np.save('/Volumes/TOURO S/Deployment1/Vectrino/npyfix/turb_'+str(ii),turb[ii])
#    np.save('/Volumes/TOURO S/Deployment1/Vectrino/npyfix/RScorr_'+str(ii),RScorr[ii])  