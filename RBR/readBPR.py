"""Sample read file for RBR BPR pressure logger .txt file"""

import datetime
import pandas as pd
import wavefuncs_rbr

filepth = ''
data = pd.read_csv(filepath, delimiter = ',', usecols = [0,5])
time = pd.to_datetime(data['Time'].values).to_pydatetime()
press = data['Sea pressure'].values

fs = 6 #Sampling frequency  
fc = 0.8 #Cutoff frequency -- might need to change later based on spectra
doffp = 1 #Pressure sensor height above bed [m]
rho = 1025 #Seawater density

depth = 1e4*press/(9.81*rho) + doffp


"""
Should break up the data into small (maybe 12 minute) chunks and then call
the wave_stats_spectra function for each chunk like: 

wavestats = wavefuncs_rbr.wave_stats_spectra(press,depth,nfft,doffp,fs,fc,rho)

Best way to do this might be to put the whole code in a loop and use the chunk_size
option in pd.read_csv 

""