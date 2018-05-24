#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 12:18:40 2018

@author: gegan
"""
import advfuncs
import copy
import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate
import pandas as pd
import wavefuncs

np.seterr(divide='ignore', invalid='ignore')

#%%
#Variables to set (input)
x_heading = 23.
vertical_orientation = 'up'
time_start = datetime.datetime(2009,2,7,11,0,0)
time_end = datetime.datetime(2009,2,21,0,6,22)
corr_min = 30.
ref_height = .01
doffu = .05
doffp = .05
rho = 1000


#Directory where all deployment files are (input)
path = '/Users/gegan/Documents/Python/Research/ADV/MooreaADVsn4611'

hdrfile = glob.glob(path + '/*.hdr')[0]
path, tail = os.path.split(hdrfile)
filename = tail[:-4]
vhdfile = filename + '.vhd'
datfile = filename + '.dat' 


#Loading in all of the data to store in adv dictionary
adv = dict()
adv['gen'] = dict()
adv['burst'] = dict()

#Deployment parameters from .hdr file
adv['gen']['start_time'] = pd.read_table(hdrfile,header = None,skiprows = 6, nrows = 1,
                   delim_whitespace = True, usecols = [4,5,6],
                   parse_dates = {'start_time':[0,1,2]}).values[0][0]
    
adv['gen']['end_time'] = pd.read_table(hdrfile,header = None,skiprows = 7, nrows = 1,
                   delim_whitespace = True, usecols = [4,5,6],
                   parse_dates = {'end_time':[0,1,2]}).values[0][0]

adv['gen']['fs'] = pd.read_table(hdrfile,header = None,skiprows = 11, nrows = 1,
                   delim_whitespace = True, usecols = [2]).values[0][0]

adv['gen']['Tburst'] = pd.read_table(hdrfile,header = None,skiprows = 13, nrows = 1,
                   delim_whitespace = True, usecols = [2]).values[0][0]

#Data from .vhd file and .dat file

datavhd = pd.read_table(vhdfile,header = None, delim_whitespace = True,usecols = range(8))
datadat = pd.read_table(datfile,header = None, delim_whitespace = True)

for ii in range(1,np.max(datavhd.loc[:,6].values)+1):
    
    adv['burst'][ii] = dict()
    
    adv['burst'][ii]['burststart'] = datetime.datetime(datavhd.values[ii-1][2],datavhd.values[ii-1][0],datavhd.values[ii-1][1],
       datavhd.values[ii-1][3],datavhd.values[ii-1][4],datavhd.values[ii-1][5])
    
    adv['burst'][ii]['burstsamples'] = datavhd.values[ii-1][7]

    #Data from .dat file
    rowidx = datadat[0] == ii

    adv['burst'][ii]['Nens'] = datadat.loc[rowidx,1].values
    
    adv['burst'][ii]['time'] = (adv['burst'][ii]['burststart'] + 
       datetime.timedelta(seconds = 1)*(1./adv['gen']['fs'])*adv['burst'][ii]['Nens'])
    
    adv['burst'][ii]['velx'] = datadat.loc[rowidx,2].values
    
    adv['burst'][ii]['vely'] = datadat.loc[rowidx,3].values
    
    adv['burst'][ii]['velz'] = datadat.loc[rowidx,4].values
    
    adv['burst'][ii]['amp1'] = datadat.loc[rowidx,5].values.astype('float')
    
    adv['burst'][ii]['amp2'] = datadat.loc[rowidx,6].values.astype('float')
    
    adv['burst'][ii]['amp3'] = datadat.loc[rowidx,7].values.astype('float')
    
    adv['burst'][ii]['corr1'] = datadat.loc[rowidx,11].values.astype('float')
    
    adv['burst'][ii]['corr2'] = datadat.loc[rowidx,12].values.astype('float')
    
    adv['burst'][ii]['corr3'] = datadat.loc[rowidx,13].values.astype('float')
    
    adv['burst'][ii]['press'] = datadat.loc[rowidx,14].values*1e4 #In pascals


#%%
#Rotating XYZ to earth coordinates
if vertical_orientation == 'up':
    roll = 180
    pitch = 0
    heading = x_heading + 90
elif vertical_orientation == 'down':
    roll = 0
    pitch = 0
    heading = x_heading-90

adv = advfuncs.xyz_enu(adv,heading,pitch,roll)


#%% Trimming the data from time_start to time_end based on when the instrument
# was actually in the water
for ii in adv['burst']:
    
    keepidx = np.intersect1d(np.array(np.where(adv['burst'][ii]['time']>time_start)),np.array(np.where(adv['burst'][ii]['time'] < time_end)))
    blen = np.max(adv['burst'][ii]['Nens'])
    
    for key in adv['burst'][ii]:
        if np.size(adv['burst'][ii][key]) == blen:
            adv['burst'][ii][key] = adv['burst'][ii][key][keepidx]

#%% Removing unrealistic velocity values and replacing with NaN
for ii in adv['burst']:
    
    #Removing values with correlations below corr_min
    corr_arr = np.stack((adv['burst'][ii]['corr1'],adv['burst'][ii]['corr2'],adv['burst'][ii]['corr3']))
    badidx =  np.where(np.nanmin(corr_arr,axis=0)<corr_min)

    adv['burst'][ii]['velx'][badidx] = np.NaN
    adv['burst'][ii]['vely'][badidx] = np.NaN
    adv['burst'][ii]['velz'][badidx] = np.NaN
    adv['burst'][ii]['press'][badidx] = np.NaN                 
    
    #Now removing values outside a few standard deviations
    badidxlow = (adv['burst'][ii]['velx'] < (np.nanmedian(adv['burst'][ii]['velx']) - 5*np.nanstd(adv['burst'][ii]['velx'])))
    
    badidxhigh = (adv['burst'][ii]['velx'] > (np.nanmedian(adv['burst'][ii]['velx']) + 5*np.nanstd(adv['burst'][ii]['velx']))) 
    
    adv['burst'][ii]['velx'][badidxlow] = np.NaN
    adv['burst'][ii]['velx'][badidxhigh] = np.NaN
    
    badidxlow = (adv['burst'][ii]['vely'] < (np.nanmedian(adv['burst'][ii]['vely']) - 5*np.nanstd(adv['burst'][ii]['vely'])))
    
    badidxhigh = (adv['burst'][ii]['vely'] > (np.nanmedian(adv['burst'][ii]['vely']) + 5*np.nanstd(adv['burst'][ii]['vely']))) 
    
    adv['burst'][ii]['vely'][badidxlow] = np.NaN
    adv['burst'][ii]['vely'][badidxhigh] = np.NaN
    
    badidxlow = (adv['burst'][ii]['velz'] < (np.nanmedian(adv['burst'][ii]['velz']) - 5*np.nanstd(adv['burst'][ii]['velz'])))
    
    badidxhigh = (adv['burst'][ii]['velz'] > (np.nanmedian(adv['burst'][ii]['velz']) + 5*np.nanstd(adv['burst'][ii]['velz']))) 
    
    adv['burst'][ii]['velz'][badidxlow] = np.NaN
    adv['burst'][ii]['velz'][badidxhigh] = np.NaN
    
    #Calculating velocity error
    adv['burst'][ii]['velerror'] = 0.005*np.sqrt(adv['burst'][ii]['velx']**2 + adv['burst'][ii]['vely']**2 + 
       adv['burst'][ii]['velz']**2) + (1./1000)
    

#%% Low-pass filtering the velocity and pressure data and placing in separate 
# structure adv_filt 
adv_filt = copy.deepcopy(adv)

for ii in adv_filt['burst']:
    adv_filt['burst'][ii]['velx'] = advfuncs.lpf(adv['burst'][ii]['velx'],
            adv['gen']['fs'],adv['gen']['fs']/30) 
    
    adv_filt['burst'][ii]['vely'] = advfuncs.lpf(adv['burst'][ii]['vely'],
            adv['gen']['fs'],adv['gen']['fs']/30) 
    
    adv_filt['burst'][ii]['velz'] = advfuncs.lpf(adv['burst'][ii]['velz'],
            adv['gen']['fs'],adv['gen']['fs']/30) 
    
    adv_filt['burst'][ii]['press'] = advfuncs.lpf(adv['burst'][ii]['press'],
            adv['gen']['fs'],adv['gen']['fs']/30) 

#%% Calculating principle axis rotation
adv['gen']['theta']=dict()
for ii in adv['burst']:
    
    adv['gen']['theta'][ii] = advfuncs.pa_theta(adv['burst'][ii]['velx'],adv['burst'][ii]['vely'])
    
    adv['burst'][ii]['velmaj'], adv['burst'][ii]['velmin'] = advfuncs.pa_rotation(
            adv['burst'][ii]['velx'],adv['burst'][ii]['vely'],adv['gen']['theta'][ii])

for ii in adv_filt['burst']:
    adv_filt['burst'][ii]['velmaj'], adv_filt['burst'][ii]['velmin'] = advfuncs.pa_rotation(
            adv_filt['burst'][ii]['velx'],adv_filt['burst'][ii]['vely'],adv['gen']['theta'][ii])
    
#%% Calculating Reynolds stress, and roughness parameters z0 and Cd   
ref_height = 0.01
vel_min = 0.01
roughness = advfuncs.get_roughness(adv,adv_filt,ref_height,vel_min,rho,doffp)


#%% Tidal analysis 

#Fill this in once you get Python t-tide up and running


#%% Wave statistics and spectra

WaveStats = dict()

for ii in adv['burst']:
    
    nfft = int(np.size(adv['burst'][ii]['velmaj']))
    
    WaveStats[ii] = wavefuncs.wave_stats_spectra(adv_filt['burst'][ii]['velmaj'],adv_filt['burst'][ii]['velmin'],adv_filt['burst'][ii]['press'],
                                         nfft,doffu,doffp,adv['gen']['fs'],adv['gen']['fs']/30,rho,0)

#%% Benilov wave/turbulence decomposition
    
Benilov = dict()

for ii in adv['burst']:
    
    nfft = int(np.size(adv['burst'][ii]['velmaj']))
    
    Benilov[ii] = wavefuncs.benilov(adv_filt['burst'][ii]['velmaj'],adv_filt['burst'][ii]['velmin'],adv_filt['burst'][ii]['velz'],
                          adv_filt['burst'][ii]['press'],nfft,doffp,adv['gen']['fs'],adv['gen']['fs']/30,rho)

#%%Merging dictionaries and calculating additional quantities 
WaveData = dict()
for ii in adv['burst']:
    WaveData[ii] = dict()
    WaveData[ii].update(WaveStats[ii])
    WaveData[ii].update(Benilov[ii])

    

#%% Some plotting to test the code

#np.save('adv_fc30_corr30.npy',adv)
#np.save('advfilt_fc30_corr30.npy',adv_filt)
#np.save('waves_fc30_corr30.npy',WaveData)
    
adv = np.load('adv_fc30_corr30.npy').item()
adv_filt = np.load('advfilt_fc30_corr30.npy').item()
waves = np.load('waves_fc30_corr30.npy').item()


plt.figure(1)
plt.plot(adv['burst'][22]['time'],adv['burst'][22]['velmaj'],label='raw')
plt.plot(adv_filt['burst'][22]['time'],adv_filt['burst'][22]['velmaj'],label='filtered')
plt.legend()
#
#
plt.figure(2)
plt.plot(adv['burst'][20]['time'],roughness[20]['upwp'])
#
plt.figure(3)
plt.plot(waves[10]['fmt'],waves[10]['SSEt'])
#
#plt.figure(4)
#plt.plot(WaveData[20]['Ust'][0:1000])
