# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:13:37 2017

@author: gegan
"""
#Reads in data from the Vectrino II profiler

import datetime
import glob
import numpy as np
import scipy.io as sio
import vecfuncs

#Constants
x_heading = 185. #Compass heading on x probe 
vertical_orientation = 'down'
corr_min = 10.  #Minimum beam correlation
snr_min = 20    #Minimum SNR
dissmethod = 'Fedd07'
wavedecomp = True
fs = 64
savepath = ''
dateidx = list(range(56,73)) #Indices in filename corresponding to unix time

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
        tstart = datetime.datetime.fromtimestamp(int(files[idx][dateidx])/1e6)
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
        tstart = datetime.datetime.fromtimestamp(int(files[idx][dateidx])/1e6)
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
    

    #Calculating principal axis rotation based on data burst. Should ideally use
    # a constant theta based on long term ADCP data
    theta =  vecfuncs.pa_theta(np.nanmean(vectrino['u'],axis=0),
            np.nanmean(vectrino['v'],axis=0))
    vectrino['theta'] = theta
    
    vectrino['velmaj'], vectrino['velmin'] = vecfuncs.pa_rotation(
            vectrino['u'],vectrino['v'],vectrino['theta'])
    

    #Calculating Reynolds stress tensor

    #Use the phase method if there is wave contamination 
    if wavedecomp:
        waveturb = vecfuncs.get_turb_waves(vectrino,fs)
    
    #Calculating other turbulence statistics, including non wave-decomposed covariances
    # and log law fitting
    turb = vecfuncs.get_turb(vectrino,fs)
        
    #Correcting with Thomas et al 2017
    RScorr = vecfuncs.noise_correction(turb[ii])
    
    #Dissipation
    diss = vecfuncs.get_dissipation(vectrino,fs,dissmethod)
    
    #Now save everything, e.g.
    np.save(savepath + 'vectrino_' + str(ii) + '.npy')
    np.save(savepath + 'waveturb_' + str(ii) + '.npy')
    np.save(savepath + 'turb_' + str(ii) + '.npy')
    np.save(savepath + 'RScorr_' + str(ii) + '.npy')
    np.save(savepath + 'diss_' + str(ii) + '.npy')