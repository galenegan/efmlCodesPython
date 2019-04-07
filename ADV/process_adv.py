#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 12:18:40 2018

@author: gegan
"""
#Packages
import os
parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

import sys
sys.path.append(parentDir + '/General/')

import advfuncs
import datetime
import glob
from mylib import naninterp
import numpy as np
import scipy.interpolate
import scipy.signal as sig
import scipy.io as sio
import pandas as pd
import wavefuncs_adv

np.seterr(divide='ignore', invalid='ignore')

#Variables to set (input)

#x_heading = 270. #Fixed ADV, don't need to rotate to ENU
#vertical_orientation = 'down'
corr_min = 30.
doffu = .15
doffp = .3
rho = 1020

#Directory where all deployment files are (input)
path = '/Users/gegan/Documents/Python/Research/Deployment1/ADVData/1316_4913'
savepath = ''

hdrfile = glob.glob(path + '/*.hdr')[0]
path, tail = os.path.split(hdrfile)
filename = tail[:-4]
vhdfile = path + '/'+ filename + '.vhd'
datfile = path + '/' + filename + '.dat' 


#Loading in all of the data to store in adv dictionary
gen = dict()

#Deployment parameters from .hdr file
gen['start_time'] = pd.read_table(hdrfile,header = None,skiprows = 6, nrows = 1,
                   delim_whitespace = True, usecols = [4],
                   parse_dates = {'start_time':[0]}).values[0][0]
    
gen['end_time'] = pd.read_table(hdrfile,header = None,skiprows = 7, nrows = 1,
                   delim_whitespace = True, usecols = [4,5,6],
                   parse_dates = {'end_time':[0,1,2]}).values[0][0]

gen['fs'] = pd.read_table(hdrfile,header = None,skiprows = 11, nrows = 1,
                   delim_whitespace = True, usecols = [2]).values[0][0]

gen['Tburst'] = pd.read_table(hdrfile,header = None,skiprows = 13, nrows = 1,
                   delim_whitespace = True, usecols = [2]).values[0][0]

np.save(savepath + 'gen.npy',gen)
#Data from .vhd file and .dat file

datavhd = pd.read_table(vhdfile,header = None, delim_whitespace = True,usecols = range(8))
datadat = pd.read_table(datfile,header = None, delim_whitespace = True)

for ii in range(len(datavhd)):
    
    adv = dict()
    
    adv['burststart'] = datetime.datetime(datavhd.values[ii][2],datavhd.values[ii][0],datavhd.values[ii][1],
       datavhd.values[ii][3],datavhd.values[ii][4],datavhd.values[ii][5])
    
    adv['burstsamples'] = datavhd.values[ii][7]

    #Data from .dat file
    rowidx = datadat[0] == ii + 1

    adv['Nens'] = datadat.loc[rowidx,1].values
    
    adv['time'] = (adv['burststart'] + 
       datetime.timedelta(seconds = 1)*(1./gen['fs'])*np.arange(1,np.size(adv['Nens'])+1))
    
    adv['velx'] = datadat.loc[rowidx,2].values
    
    adv['vely'] = datadat.loc[rowidx,3].values
    
    adv['velz'] = datadat.loc[rowidx,4].values
    
    adv['amp1'] = datadat.loc[rowidx,5].values.astype('float')
    
    adv['amp2'] = datadat.loc[rowidx,6].values.astype('float')
    
    adv['amp3'] = datadat.loc[rowidx,7].values.astype('float')
    
    adv['corr1'] = datadat.loc[rowidx,11].values.astype('float')
    
    adv['corr2'] = datadat.loc[rowidx,12].values.astype('float')
    
    adv['corr3'] = datadat.loc[rowidx,13].values.astype('float')
    
    adv['press'] = datadat.loc[rowidx,14].values


    ##Rotating XYZ to earth coordinates
    #if vertical_orientation == 'up':
    #    roll = 180
    #    pitch = 0
    #    heading = x_heading + 90
    #elif vertical_orientation == 'down':
    #    roll = 0
    #    pitch = 0
    #    heading = x_heading-90
    #
    #adv = advfuncs.xyz_enu(adv,heading,pitch,roll)


    #Removing values with correlations below corr_min
    corr_arr = np.stack((adv['corr1'],adv['corr2'],adv['corr3']))
    badidx =  np.where(np.nanmin(corr_arr,axis=0)<corr_min)

    adv['velx'][badidx] = np.NaN
    adv['vely'][badidx] = np.NaN
    adv['velz'][badidx] = np.NaN
    adv['press'][badidx] = np.NaN                 
    

    #Now removing values outside a few standard deviations
    badidxlow = (adv['velx'] < (np.nanmedian(adv['velx']) - 5*np.nanstd(adv['velx'])))
    
    badidxhigh = (adv['velx'] > (np.nanmedian(adv['velx']) + 5*np.nanstd(adv['velx']))) 
    
    adv['velx'][badidxlow] = np.NaN
    adv['velx'][badidxhigh] = np.NaN
    
    badidxlow = (adv['vely'] < (np.nanmedian(adv['vely']) - 5*np.nanstd(adv['vely'])))
    
    badidxhigh = (adv['vely'] > (np.nanmedian(adv['vely']) + 5*np.nanstd(adv['vely']))) 
    
    adv['vely'][badidxlow] = np.NaN
    adv['vely'][badidxhigh] = np.NaN
    
    badidxlow = (adv['velz'] < (np.nanmedian(adv['velz']) - 5*np.nanstd(adv['velz'])))
    
    badidxhigh = (adv['velz'] > (np.nanmedian(adv['velz']) + 5*np.nanstd(adv['velz']))) 
    
    adv['velz'][badidxlow] = np.NaN
    adv['velz'][badidxhigh] = np.NaN
    
    #Removing values above a certain magnitude--adjust for each data set.
    badmag = 0.5
    adv['velx'][np.abs(adv['velx'])>badmag] = np.NaN
    adv['vely'][np.abs(adv['vely'])>badmag] = np.NaN
    adv['velz'][np.abs(adv['velz'])>badmag] = np.NaN
    
    

    #%% Low-pass filtering the velocity and pressure data and placing in separate 
    # structure adv_filt 

    fc = 0.1

    adv['velxfilt'] = advfuncs.lpf(adv['velx'],
            gen['fs'],fc) 
    
    adv['velyfilt'] = advfuncs.lpf(adv['vely'],
            gen['fs'],fc) 
    
    adv['velzfilt'] = advfuncs.lpf(adv['velz'],
            gen['fs'],fc) 
    
    adv['pressfilt'] = advfuncs.lpf(adv['press'],
            gen['fs'],fc) 

    #%% Calculating principle axis rotation
 
    adv['theta'] = advfuncs.pa_theta(adv['velx'],adv['vely'])
    
    adv['velmaj'], adv['velmin'] = advfuncs.pa_rotation(
            adv['velx'],adv['vely'],adv['theta'])

    adv['velmajfilt'], adv['velminfilt'] = advfuncs.pa_rotation(
            adv['velxfilt'],adv['velyfilt'],adv['theta'])
    

    #%% Wave statistics and spectra

    fcw = gen['fs']/8
    nfft = 14*60*gen['fs']/3


    WaveStats = wavefuncs_adv.wave_stats_spectra(adv['velmaj'],adv['velmin'],
                adv['press'],nfft,doffu,doffp,gen['fs'],fcw,rho,0)
    
    # Benilov wave/turbulence decomposition  

    Benilov = wavefuncs_adv.benilov(adv['velmaj'],adv['velmin'],adv['velz'],
              adv['press'],doffp,gen['fs'],gen['fs']/8,rho)

    #Phase method wave/turbulence decomposition
    Phase = wavefuncs_adv.bm_phase(adv,gen['fs'])
    
    np.save(savepath + 'adv_' + str(ii) + '.npy',adv)
    np.save(savepath + 'benilov_' + str(ii) + '.npy',Benilov)
    np.save(savepath + 'phase_' + str(ii) + '.npy',Phase)
    np.save(savepath + 'wavestats_' + str(ii) + '.npy',WaveStats)

