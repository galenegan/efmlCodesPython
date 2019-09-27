#Packages
import advfuncs
import datetime
import glob
import numpy as np
import os
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')

#Variables to set (input)

#x_heading = 270. #Only specify if it's a flex head. Don't need to rotate for fixed
#vertical_orientation = 'down'
corr_min = 30. #Minimum beam correlation
doffu = .15 #Sampling height
doffp = .3 #Pressure sensor height
rho = 1020 #Water density

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


    ##Rotating XYZ to earth coordinates -- do this if non-fixed ADV head
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
    K = 4
    badidxlow = (adv['velx'] < (np.nanmedian(adv['velx']) - K*np.nanstd(adv['velx'])))
    
    badidxhigh = (adv['velx'] > (np.nanmedian(adv['velx']) + K*np.nanstd(adv['velx']))) 
    
    adv['velx'][badidxlow] = np.NaN
    adv['velx'][badidxhigh] = np.NaN
    
    badidxlow = (adv['vely'] < (np.nanmedian(adv['vely']) - K*np.nanstd(adv['vely'])))
    
    badidxhigh = (adv['vely'] > (np.nanmedian(adv['vely']) + K*np.nanstd(adv['vely']))) 
    
    adv['vely'][badidxlow] = np.NaN
    adv['vely'][badidxhigh] = np.NaN
    
    badidxlow = (adv['velz'] < (np.nanmedian(adv['velz']) - K*np.nanstd(adv['velz'])))
    
    badidxhigh = (adv['velz'] > (np.nanmedian(adv['velz']) + K*np.nanstd(adv['velz']))) 
    
    adv['velz'][badidxlow] = np.NaN
    adv['velz'][badidxhigh] = np.NaN
    
    #Removing values above a certain magnitude--adjust for each data set
    badmag = 0.5
    adv['velx'][np.abs(adv['velx'])>badmag] = np.NaN
    adv['vely'][np.abs(adv['vely'])>badmag] = np.NaN
    adv['velz'][np.abs(adv['velz'])>badmag] = np.NaN
    
    

    #%% Calculating principle axis rotation. Should ideally use constant theta 
    #from ADCP data
 
    adv['theta'] = advfuncs.pa_theta(adv['velx'],adv['vely'])
    
    adv['velmaj'], adv['velmin'] = advfuncs.pa_rotation(
            adv['velx'],adv['vely'],adv['theta'])

    #%% Wave statistics and spectra

    fcw = gen['fs']/8
    nfft = 14*60*gen['fs']/3

    WaveStats = advfuncs.wave_stats_spectra(adv['velmaj'],adv['velmin'],
                adv['press'],nfft,doffu,doffp,gen['fs'],fcw,rho,0)
    
    # Benilov wave/turbulence decomposition  
    Benilov = advfuncs.benilov(adv['velmaj'],adv['velmin'],adv['velz'],
              adv['press'],doffp,gen['fs'],gen['fs']/8,rho)

    
    np.save(savepath + 'adv_' + str(ii) + '.npy',adv)
    np.save(savepath + 'benilov_' + str(ii) + '.npy',Benilov)
    np.save(savepath + 'wavestats_' + str(ii) + '.npy',WaveStats)