#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:13:37 2017

@author: gegan
"""
#Reads in data from the Vectrino II profiler

#Packages
import copy
import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import vecfuncs

#Constants
x_heading = 0.
vertical_orientation = 'down'
time_start = datetime.datetime(2018,2,7,11,0,0)
time_end = datetime.datetime(2018,2,21,0,6,22)
corr_min = 10.
snr_min = 20
lp_cutoff = 25.
#%%Loading in data and assigning to variables
files = glob.glob('*.mat')
Nburst = np.size(files)
burstnums = np.arange(Nburst)

vectrino = dict()
vectrino['burst'] = dict()
for ii in burstnums:
    vectrino['burst'][ii] = dict()

vectrino['gen'] = dict()

vectrino['gen']['fs'] = 32

for ii in range(Nburst):
    
    matdata = sio.loadmat(files[ii],matlab_compatible=True)
    Data = matdata[list(matdata.keys())[3]]
    Config = matdata[list(matdata.keys())[4]]
 
    #Time 
    tstart = datetime.datetime.fromtimestamp(int(files[ii][0:13])/1000.) 
    t = tstart+datetime.timedelta(seconds=1)*Data['Profiles_HostTime'].item().squeeze().T   
    vectrino['burst'][ii]['time'] = t
    
    #The rest of the data
    vectrino['burst'][ii]['u'] = Data['Profiles_VelX'].item().T
    vectrino['burst'][ii]['v'] = Data['Profiles_VelY'].item().T
    vectrino['burst'][ii]['w1'] = Data['Profiles_VelZ1'].item().T
    vectrino['burst'][ii]['w2'] = Data['Profiles_VelZ2'].item().T
    
    vectrino['burst'][ii]['corr1'] = Data['Profiles_CorBeam1'].item().T
    vectrino['burst'][ii]['corr2'] = Data['Profiles_CorBeam2'].item().T
    vectrino['burst'][ii]['corr3'] = Data['Profiles_CorBeam3'].item().T
    vectrino['burst'][ii]['corr4'] = Data['Profiles_CorBeam4'].item().T
    
    vectrino['burst'][ii]['amp1'] = Data['Profiles_AmpBeam1'].item().T
    vectrino['burst'][ii]['amp2'] = Data['Profiles_AmpBeam2'].item().T
    vectrino['burst'][ii]['amp3'] = Data['Profiles_AmpBeam3'].item().T
    vectrino['burst'][ii]['amp4'] = Data['Profiles_AmpBeam4'].item().T
    
    vectrino['burst'][ii]['snr1'] = Data['Profiles_SNRBeam1'].item().T
    vectrino['burst'][ii]['snr2'] = Data['Profiles_SNRBeam2'].item().T
    vectrino['burst'][ii]['snr3'] = Data['Profiles_SNRBeam3'].item().T
    vectrino['burst'][ii]['snr4'] = Data['Profiles_SNRBeam4'].item().T


#%% Rotating XYZ to earth coordinates
if vertical_orientation == 'up':
    roll = 180
    pitch = 0
    heading = x_heading + 90
elif vertical_orientation == 'down':
    roll = 0
    pitch = 0
    heading = x_heading-90

vectrino = vecfuncs.xyz_enu(vectrino,heading,pitch,roll)


#%% Removing unrealistic velocity values and replacing with NaN
for ii in vectrino['burst']:
    
    #Removing values with correlations below corr_min and snr below snr_min
    badidx1 = (vectrino['burst'][ii]['corr1']<corr_min) | (vectrino['burst'][ii]['snr1']<snr_min)
    badidx2 = (vectrino['burst'][ii]['corr2']<corr_min) | (vectrino['burst'][ii]['snr2']<snr_min)
    badidx3 = (vectrino['burst'][ii]['corr3']<corr_min) | (vectrino['burst'][ii]['snr3']<snr_min)
    badidx4 = (vectrino['burst'][ii]['corr4']<corr_min) | (vectrino['burst'][ii]['snr4']<snr_min)
    
    
    vectrino['burst'][ii]['u'][badidx1] = np.NaN
    vectrino['burst'][ii]['v'][badidx2] = np.NaN
    vectrino['burst'][ii]['w1'][badidx3] = np.NaN
    vectrino['burst'][ii]['w2'][badidx4] = np.NaN      

    del badidx1, badidx2, badidx3, badidx4           
    
    #Calculating velocity error
    vectrino['burst'][ii]['velerror'] = 0.005*np.sqrt(vectrino['burst'][ii]['u']**2 + vectrino['burst'][ii]['v']**2 + 
       vectrino['burst'][ii]['w1']**2) + (1./1000)
    
#%% Low-pass filtering the velocity and pressure data and placing in separate 
# structure adv_filt 
vectrino_filt = copy.deepcopy(vectrino)

for ii in vectrino_filt['burst']:
    vectrino_filt['burst'][ii]['u'] = vecfuncs.lpf(vectrino['burst'][ii]['u'],
            vectrino['gen']['fs'],vectrino['gen']['fs']/lp_cutoff) 
    
    vectrino_filt['burst'][ii]['v'] = vecfuncs.lpf(vectrino['burst'][ii]['v'],
            vectrino['gen']['fs'],vectrino['gen']['fs']/lp_cutoff) 
    
    vectrino_filt['burst'][ii]['w1'] = vecfuncs.lpf(vectrino['burst'][ii]['w1'],
            vectrino['gen']['fs'],vectrino['gen']['fs']/lp_cutoff) 
    
    vectrino_filt['burst'][ii]['w2'] = vecfuncs.lpf(vectrino['burst'][ii]['w2'],
            vectrino['gen']['fs'],vectrino['gen']['fs']/lp_cutoff) 
    
#%% Calculating principle axis rotation
vectrino['gen']['theta']=dict()
for ii in vectrino['burst']:
    
    vectrino['gen']['theta'][ii] = vecfuncs.pa_theta(np.mean(vectrino['burst'][ii]['u'],axis=0),
            np.mean(vectrino['burst'][ii]['v'],axis=0))
    
    vectrino['burst'][ii]['velmaj'], vectrino['burst'][ii]['velmin'] = vecfuncs.pa_rotation(
            vectrino['burst'][ii]['u'],vectrino['burst'][ii]['v'],vectrino['gen']['theta'][ii])

for ii in vectrino_filt['burst']:
    vectrino_filt['burst'][ii]['velmaj'], vectrino_filt['burst'][ii]['velmin'] = vecfuncs.pa_rotation(
            vectrino_filt['burst'][ii]['u'],vectrino_filt['burst'][ii]['v'],vectrino['gen']['theta'][ii])
    
#%% Some plotting to test the code

#plt.figure(1)
#plt.plot(vectrino['burst'][1]['u'][2,:],label='raw')
#plt.plot(vectrino_filt['burst'][1]['u'][2,:],label='filtered')
#plt.legend()

#plt.figure(2)
#for ii in range(np.size(vectrino['burst'][1]['u'],axis=1)):
#    plt.plot(vectrino['burst'][1]['u'][:,ii],label='raw')
#    plt.plot(vectrino_filt['burst'][1]['u'][:,ii],label='raw')
#    plt.pause(.01)
#    plt.clf()
#plt.show()