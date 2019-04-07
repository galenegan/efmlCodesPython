# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:13:37 2017

@author: gegan
"""
#Reads in data from the Vectrino II profiler

#Packages
import sys
sys.path.append('../General/')

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
files = glob.glob('/Volumes/TOURO S/Deployment1/Vectrino/vectrino_data_mat/*.mat')
files.sort()
Nfiles = np.size(files)

burstnums = []

for file in files:
    bnum = file.split('-')[1].split('.data')[0]
    burstnums = np.concatenate((burstnums,[int(bnum)]))
burstnums = np.unique(burstnums,return_index = True)
burstnum = burstnums[0].astype(int)
burstind = burstnums[1]


#Initializing dicts
vectrino = dict()
vectrino_filt = dict()
waveturb = dict()
turb = dict()
RScorr = dict()

for ii in burstnum:
    vectrino[ii] = dict()
    vectrino_filt[ii] = dict()
    turb[ii] = dict()
    waveturb[ii] = dict()
    RScorr[ii] = dict()

for ii in range(395,len(burstind)-1):
    
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
        vectrino[ii]['time'] = t
        
        #Bottom distance and vertical range
        bottom_dist = np.concatenate((Data1['BottomCheck_BottomDistance'].item().squeeze(),
                                      Data2['BottomCheck_BottomDistance'].item().squeeze()))
    
        if np.all(bottom_dist == 0):
            vectrino[ii]['bot'] = 0.06
        else:
            vectrino[ii]['bot'] = np.nanmean(bottom_dist[bottom_dist!=0])
    
        vrange = Data1['Profiles_Range'].item().T.squeeze()
        vectrino[ii]['z']   = -vrange + vectrino[ii]['bot']
    
        #The rest of the data
        vectrino[ii]['u'] = np.concatenate((Data1['Profiles_VelX'].item().squeeze(),Data2['Profiles_VelX'].item().squeeze())).T
        vectrino[ii]['v'] = np.concatenate((Data1['Profiles_VelY'].item().squeeze(),Data2['Profiles_VelY'].item().squeeze())).T
        vectrino[ii]['w1'] = np.concatenate((Data1['Profiles_VelZ1'].item().squeeze(),Data2['Profiles_VelZ1'].item().squeeze())).T
        vectrino[ii]['w2'] = np.concatenate((Data1['Profiles_VelZ2'].item().squeeze(),Data2['Profiles_VelZ2'].item().squeeze())).T
        
        vectrino[ii]['corr1'] = np.concatenate((Data1['Profiles_CorBeam1'].item().squeeze(),Data2['Profiles_CorBeam1'].item().squeeze())).T
        vectrino[ii]['corr2'] = np.concatenate((Data1['Profiles_CorBeam2'].item().squeeze(),Data2['Profiles_CorBeam2'].item().squeeze())).T
        vectrino[ii]['corr3'] = np.concatenate((Data1['Profiles_CorBeam3'].item().squeeze(),Data2['Profiles_CorBeam3'].item().squeeze())).T
        vectrino[ii]['corr4'] = np.concatenate((Data1['Profiles_CorBeam4'].item().squeeze(),Data2['Profiles_CorBeam4'].item().squeeze())).T
        
        vectrino[ii]['amp1'] = np.concatenate((Data1['Profiles_AmpBeam1'].item().squeeze(),Data2['Profiles_AmpBeam1'].item().squeeze())).T
        vectrino[ii]['amp2'] = np.concatenate((Data1['Profiles_AmpBeam2'].item().squeeze(),Data2['Profiles_AmpBeam2'].item().squeeze())).T
        vectrino[ii]['amp3'] = np.concatenate((Data1['Profiles_AmpBeam3'].item().squeeze(),Data2['Profiles_AmpBeam3'].item().squeeze())).T
        vectrino[ii]['amp4'] = np.concatenate((Data1['Profiles_AmpBeam4'].item().squeeze(),Data2['Profiles_AmpBeam4'].item().squeeze())).T
    
        vectrino[ii]['snr1'] = np.concatenate((Data1['Profiles_SNRBeam1'].item().squeeze(),Data2['Profiles_SNRBeam1'].item().squeeze())).T
        vectrino[ii]['snr2'] = np.concatenate((Data1['Profiles_SNRBeam2'].item().squeeze(),Data2['Profiles_SNRBeam2'].item().squeeze())).T
        vectrino[ii]['snr3'] = np.concatenate((Data1['Profiles_SNRBeam3'].item().squeeze(),Data2['Profiles_SNRBeam3'].item().squeeze())).T
        vectrino[ii]['snr4'] = np.concatenate((Data1['Profiles_SNRBeam4'].item().squeeze(),Data2['Profiles_SNRBeam4'].item().squeeze())).T
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
        vectrino[ii]['time'] = t
        
        #Bottom distance and vertical range
        bottom_dist = Data1['BottomCheck_BottomDistance'].item().squeeze()
    
        if np.all(bottom_dist == 0):
            vectrino[ii]['bot'] = 0.06
        else:
            vectrino[ii]['bot'] = np.nanmean(bottom_dist[bottom_dist!=0])
    
        vrange = Data1['Profiles_Range'].item().T.squeeze()
        vectrino[ii]['z']   = -vrange + vectrino[ii]['bot']
    
        #The rest of the data
        vectrino[ii]['u'] = Data1['Profiles_VelX'].item().squeeze().T
        vectrino[ii]['v'] = Data1['Profiles_VelY'].item().squeeze().T
        vectrino[ii]['w1'] = Data1['Profiles_VelZ1'].item().squeeze().T
        vectrino[ii]['w2'] = Data1['Profiles_VelZ2'].item().squeeze().T
        
        vectrino[ii]['corr1'] = Data1['Profiles_CorBeam1'].item().squeeze().T
        vectrino[ii]['corr2'] = Data1['Profiles_CorBeam2'].item().squeeze().T
        vectrino[ii]['corr3'] = Data1['Profiles_CorBeam3'].item().squeeze().T
        vectrino[ii]['corr4'] = Data1['Profiles_CorBeam4'].item().squeeze().T
        
        vectrino[ii]['amp1'] = Data1['Profiles_AmpBeam1'].item().squeeze().T
        vectrino[ii]['amp2'] = Data1['Profiles_AmpBeam2'].item().squeeze().T
        vectrino[ii]['amp3'] = Data1['Profiles_AmpBeam3'].item().squeeze().T
        vectrino[ii]['amp4'] = Data1['Profiles_AmpBeam4'].item().squeeze().T
    
        vectrino[ii]['snr1'] = Data1['Profiles_SNRBeam1'].item().squeeze().T
        vectrino[ii]['snr2'] = Data1['Profiles_SNRBeam2'].item().squeeze().T
        vectrino[ii]['snr3'] = Data1['Profiles_SNRBeam3'].item().squeeze().T
        vectrino[ii]['snr4'] = Data1['Profiles_SNRBeam4'].item().squeeze().T
    # Rotating XYZ to earth coordinates
    if vertical_orientation == 'up':
        roll = 180
        pitch = 0
        heading = x_heading + 90
    elif vertical_orientation == 'down':
        roll = 0
        pitch = 0
        heading = x_heading-90
    
    vectrino[ii] = vecfuncs.xyz_enu(vectrino[ii],heading,pitch,roll)
    
    # Filtering out bad data and trimming below the bed
    #import matlab.engine
    #eng = matlab.engine.start_matlab()
    #

    
#    #Despiking with kde method at boundaries
#    for jj in (0,29):
#    
#        u = list(vectrino[ii]['u'][jj,:])
#        v = list(vectrino[ii]['v'][jj,:])
#        w1 = list(vectrino[ii]['w1'][jj,:])
#        w2 = list(vectrino[ii]['w2'][jj,:])
#        
#        unew = eng.despikeADV(matlab.double(u),.01,.01)
#        vnew = eng.despikeADV(matlab.double(v),.01,.01)
#        w1new = eng.despikeADV(matlab.double(w1),.01,.01)
#        w2new = eng.despikeADV(matlab.double(w2),.01,.01)
#    
#        vectrino[ii]['u'][jj,:] = np.asarray(unew).squeeze()
#        vectrino[ii]['v'][jj,:] = np.asarray(vnew).squeeze()
#        vectrino[ii]['w1'][jj,:] = np.asarray(w1new).squeeze()
#        vectrino[ii]['w2'][jj,:] = np.asarray(w2new).squeeze()
    
    #NaN below the bed
    trimbins = (vectrino[ii]['z'] < 0)
    
    
    vectrino[ii]['u'][trimbins,:] = np.NaN
    vectrino[ii]['v'][trimbins,:] = np.NaN
    vectrino[ii]['w1'][trimbins,:] = np.NaN
    vectrino[ii]['w2'][trimbins,:] = np.NaN
    
    vectrino[ii]['corr1'][trimbins,:] = np.NaN
    vectrino[ii]['corr2'][trimbins,:] = np.NaN
    vectrino[ii]['corr3'][trimbins,:] = np.NaN
    vectrino[ii]['corr4'][trimbins,:] = np.NaN
    
    vectrino[ii]['amp1'][trimbins,:] = np.NaN
    vectrino[ii]['amp2'][trimbins,:] = np.NaN
    vectrino[ii]['amp3'][trimbins,:] = np.NaN
    vectrino[ii]['amp4'][trimbins,:] = np.NaN
    
    vectrino[ii]['snr1'][trimbins,:] = np.NaN
    vectrino[ii]['snr2'][trimbins,:] = np.NaN
    vectrino[ii]['snr3'][trimbins,:] = np.NaN
    vectrino[ii]['snr4'][trimbins,:] = np.NaN
    
#    
    #Removing values with correlations below corr_min and snr below snr_min
    badidx1 = (vectrino[ii]['corr1']<corr_min) | (vectrino[ii]['snr1']<snr_min)
    badidx2 = (vectrino[ii]['corr2']<corr_min) | (vectrino[ii]['snr2']<snr_min)
    badidx3 = (vectrino[ii]['corr3']<corr_min) | (vectrino[ii]['snr3']<snr_min)
    badidx4 = (vectrino[ii]['corr4']<corr_min) | (vectrino[ii]['snr4']<snr_min)
    
    
    vectrino[ii]['u'][badidx1] = np.NaN
    vectrino[ii]['v'][badidx2] = np.NaN
    vectrino[ii]['w1'][badidx3] = np.NaN
    vectrino[ii]['w2'][badidx4] = np.NaN      
    

    #Calculating velocity error
    vectrino[ii]['velerror'] = 0.005*np.sqrt(vectrino[ii]['u']**2 + vectrino[ii]['v']**2 + 
       vectrino[ii]['w1']**2) + (1./1000)
    
    
    #Calculating principal axis rotation
    theta =  vecfuncs.pa_theta(np.nanmean(vectrino[ii]['u'],axis=0),
            np.nanmean(vectrino[ii]['v'],axis=0))
    vectrino[ii]['theta'] = theta
    
    vectrino[ii]['velmaj'], vectrino[ii]['velmin'] = vecfuncs.pa_rotation(
            vectrino[ii]['u'],vectrino[ii]['v'],vectrino[ii]['theta'])
    
#    #Calculating SSC
#    snr = np.sqrt(vectrino[ii]['snr1'][10,:]**2 + vectrino[ii]['snr2'][10,:]**2 + 
#           vectrino[ii]['snr3'][10,:]**2 + vectrino[ii]['snr4'][10,:]**2)
#    logssc = -0.62963165 + 0.03362257*snr
#    vectrino[ii]['SSC'] = 10**(logssc)
    
   
    
    #Low-pass filtering the velocity and pressure data and placing in separate 
    #structure 
    
    fc = 1./30 #Cutoff frequency for fourier and butterworth filters
    filt_kernel =  (3,9) #kernel for median filter

    # Method 1--fourier filter
    if filtstyle == 'fourier':

        vectrino_filt[ii]['u'] = vecfuncs.lpf(vectrino[ii]['u'],
                fs,fc) 
        
        vectrino_filt[ii]['v'] = vecfuncs.lpf(vectrino[ii]['v'],
                fs,fc) 
        
        vectrino_filt[ii]['w1'] = vecfuncs.lpf(vectrino[ii]['w1'],
                fs,fc) 
        
        vectrino_filt[ii]['w2'] = vecfuncs.lpf(vectrino[ii]['w2'],
                fs,fc) 

    #Method 2: median filter 
    elif filtstyle == 'median':
    
        #filt_kernel = 9
        vectrino_filt[ii]['u'] = scipy.ndimage.filters.median_filter(vectrino[ii]['u'],
                     size = filt_kernel,mode = 'nearest')
        vectrino_filt[ii]['v'] = scipy.ndimage.filters.median_filter(vectrino[ii]['v'],
                     size = filt_kernel,mode = 'nearest')
        vectrino_filt[ii]['w1'] = scipy.ndimage.filters.median_filter(vectrino[ii]['w1'],
                     size = filt_kernel,mode = 'nearest')
        vectrino_filt[ii]['w2'] = scipy.ndimage.filters.median_filter(vectrino[ii]['w2'],
                     size = filt_kernel,mode = 'nearest')
        vectrino_filt[ii]['velmaj'] = scipy.ndimage.filters.median_filter(vectrino[ii]['velmaj'],
                     size = filt_kernel,mode = 'nearest')
        vectrino_filt[ii]['velmin'] = scipy.ndimage.filters.median_filter(vectrino[ii]['velmin'],
                     size = filt_kernel,mode = 'nearest')
        vectrino_filt[ii]['SSC'] = scipy.ndimage.filters.median_filter(vectrino[ii]['SSC'],
                     size = filt_kernel[1],mode = 'nearest')
        
    elif filtstyle == 'butter':
        Wn = fc/(fs/2)
        b,a = sig.butter(2,Wn,btype = 'low')
        vectrino_filt[ii]['u'] = np.empty(np.shape(vectrino[ii]['u']))
        vectrino_filt[ii]['v'] = np.empty(np.shape(vectrino[ii]['v']))
        vectrino_filt[ii]['w1'] = np.empty(np.shape(vectrino[ii]['w1']))
        vectrino_filt[ii]['w2'] = np.empty(np.shape(vectrino[ii]['w2']))
        vectrino_filt[ii]['velmaj'] = np.empty(np.shape(vectrino[ii]['velmaj']))
        vectrino_filt[ii]['velmin'] = np.empty(np.shape(vectrino[ii]['velmin']))
        
        for jj in range(np.size(vrange)):
            u = mylib.naninterp(vectrino[ii]['u'][jj,:])
            v = mylib.naninterp(vectrino[ii]['v'][jj,:])
            w1 = mylib.naninterp(vectrino[ii]['w1'][jj,:])
            w2 = mylib.naninterp(vectrino[ii]['w2'][jj,:])
            velmaj = mylib.naninterp(vectrino[ii]['velmaj'][jj,:])
            velmin = mylib.naninterp(vectrino[ii]['velmin'][jj,:])
            vectrino_filt[ii]['u'][jj,:] = sig.filtfilt(b,a,u)
            vectrino_filt[ii]['v'][jj,:] = sig.filtfilt(b,a,v)
            vectrino_filt[ii]['w1'][jj,:] = sig.filtfilt(b,a,w1)
            vectrino_filt[ii]['w2'][jj,:] = sig.filtfilt(b,a,w2)
            vectrino_filt[ii]['velmaj'][jj,:] = sig.filtfilt(b,a,velmaj)
            vectrino_filt[ii]['velmin'][jj,:] = sig.filtfilt(b,a,velmin)

    
    # Calculating Reynolds stress tensor

    #Can choose either phase method or benilov method. Benilov requires pressure data. 
    if wavedecomp:
        method = 'phase'
        waveturb[ii] = vecfuncs.get_turb_waves(vectrino[ii],fs,method)
    
    #Can change filtstyle to nofilt to calculate reynolds stress based solely on covariance (i.e. one data point per burst per cell)
   
    #Calculating other turbulence statistics, including non wave-decomposed covariances
    # and log law fitting
    filt_style_cov = 'nofilt'
    turb[ii] = vecfuncs.get_turb(vectrino[ii],vectrino_filt[ii],fs,fc,filt_kernel,filt_style_cov)
        
#    #Correcting with Thomas et al 2017. Only do this if reynolds stress calculation
#    #gives one value per burst (i.e. wave turbulence decomposition, or nofilt covariance method)
    RScorr[ii] = vecfuncs.noise_correction(turb[ii])
    
    #Saving the data...there's a bug in pickle that doesn't allow saving the entire thing 
        #because it's too big. So saving each burst individually
#    
    np.save('/Volumes/TOURO S/Deployment1/Vectrino/npyfix/vectrino_'+str(ii),vectrino[ii])
    np.save('/Volumes/TOURO S/Deployment1/Vectrino/npyfix/vectrinofilt_'+str(ii),vectrino_filt[ii])
    np.save('/Volumes/TOURO S/Deployment1/Vectrino/npyfix/waveturb_'+str(ii),waveturb[ii])
    np.save('/Volumes/TOURO S/Deployment1/Vectrino/npyfix/turb_'+str(ii),turb[ii])
    np.save('/Volumes/TOURO S/Deployment1/Vectrino/npyfix/RScorr_'+str(ii),RScorr[ii])    

   

#%% Test plotting
#plt.figure(1)
#plt.plot(RScorr[1]['var13']/RScorr[1]['w1w2'],vectrino[1]['z'])
#plt.figure(2)
#plt.plot(turb[0]['uv_wave'],vectrino[0]['z'],'C0-')
#plt.plot(turb[0]['uv'],vectrino[0]['z'],'C1--')


#%% Loading in the data
#    
#for ii in range(14):
#    np.load('./npyfiles_0709/vectrino_'+str(ii)+'.npy')

##%% Power spectra
#    
#params = {
#   'axes.labelsize': 16,
#   'font.size': 16,
#   'legend.fontsize': 12,
#   'xtick.labelsize': 16,
#   'ytick.labelsize': 16,
#   'text.usetex': False,
#   'font.family': 'serif',
#   'font.serif': 'Times'
#   }
#
#velspec = mylib.naninterp(vectrino[1]['velmaj'][10,:])
#
#f,Puu = sig.welch(velspec,fs = 64,window = 'hamming',nperseg = 64*400)
#
#mask = np.where((f>0.1) & (f<7))
#fplot = f[mask]
#plt.loglog(f,Puu,'C0-')
#plt.loglog(fplot,4e-5*fplot**(-5/3),'C1-')
#plt.xlabel(r'$f$ [Hz]')
#plt.ylabel(r'$S_{u u}$ [m$^2$ s$^{-2}$ /Hz]')
#plt.title('Power Spectral Density')
#plt.rcParams.update(params)
#    
#%% Some plotting to test the code
#
        
 
##Testing the log-law fit    
#gradloc = 17
#logind = np.arange(0,gradloc)
#vslind = np.arange(gradloc-2,gradloc+3)   
#     
#plt.figure(1)
#plt.plot(np.nanmean(vectrino['velmaj'],axis = 1),vectrino['z'],'ko')
#plt.plot(np.log(vectrino['z'][logind]/turb['z0_fit_log'])*turb['ustar_fit_log']/0.41,vectrino['z'][logind],'r-')
#plt.plot(vectrino['z'][vslind]*turb['ustar_fit_vsl']**2/1e-6 + turb['vsl_intercept'],vectrino['z'][vslind],'r--')

#Z,T = np.meshgrid(vectrino[1]['z'],vectrino[1]['time'])
#
#Z = Z.T
#T = T.T
#
#timevec = pd.date_range(vectrino[]['time'][0],vectrino[2]['time'][-1],freq = '2min' )
#
#fig, ax = plt.subplots()
#plt.contourf(T,Z,vectrino[1]['velmaj'],20)
#
#
#c = plt.colorbar()
#c.ax.set_title('u [m/s]',pad = 12)
#plt.clim(-0.2,0)
#plt.ylabel('MAB')
#plt.xlabel('Time of Day, 06/01/18')
#plt.title('Mean Currents')
#
#plt.xticks(timevec)
#xfmt = md.DateFormatter('%H:%M')
#ax.xaxis.set_major_formatter(xfmt)
#fig.autofmt_xdate()
#plt.rcParams.update(params)

    
#plt.rc('text',usetex = True)
##plt.figure(1)
##plt.plot(vectrino[1]['u'][2,:],label='raw')
##plt.plot(vectrino_filt[1]['u'][2,:],label='filtered')
##plt.legend()
#
#plt.figure(2)
#
#for ii in range(np.size(vectrino[0]['u'],axis=1)):
#    plt.plot(vectrino[0]['u'][:,ii],vectrino[0]['z'],label='raw')
#    plt.xlim(-0.01,0.1)
#    plt.plot(test[:,ii],vectrino[1]['z'],label='filt')
#    plt.pause(.01)
#    plt.clf()
#plt.show()
#
#plt.figure(3)
#plt.plot(roughness[2]['upw1p'][:,100],vectrino[2]['z'])
#
#plt.figure(4)
#plt.plot(vectrino[2]['u'][:,100],vectrino[2]['z'])

#plt.plot(np.nanmean(vectrino['velmaj'][vectrino['z']>=0],axis = 1),vectrino['z'][vectrino['z']>=0])
#plt.xlabel('u [m/s]')
#plt.ylabel('MAB')
#plt.title('Mean velocity profile')
#plt.rcParams.update(params)



#timevec = pd.date_range(vectrino[1]['time'][0],vectrino[1]['time'][-1],freq = '2min' )
#
#fig, ax = plt.subplots()
#
#plt.plot(vectrino[1]['time'],np.nanmean(vectrino[1]['velmaj'],axis = 0))
#plt.xlabel('Time of day 06/01/2018')
#plt.ylabel('U [m/s]')
#plt.title('Depth-averaged velocity')
#
#
#plt.xticks(timevec)
#xfmt = md.DateFormatter('%H:%M')
#ax.xaxis.set_major_formatter(xfmt)
#fig.autofmt_xdate()
#
#plt.rcParams.update(params)