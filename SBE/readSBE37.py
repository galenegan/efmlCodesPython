#read sbe37
import numpy as np
import datetime
import pandas as pd


#%% Importing CTD data 

filepath = ''
ctd = pd.read_table(filepath,delimiter = ',', names = ['temp','sal','date','time'],
                    skip_blank_lines = False,header = 59,usecols = [0,3,4,5],
                    parse_dates = [[2,3]])

ctd_time = pd.to_datetime(ctd['date_time'].values).to_pydatetime()
sal = ctd['sal'].values
temp = ctd['temp'].values

tstart = datetime.datetime(2018,7,18,0,0,0)
tend = datetime.datetime(2018,8,15,0,0,0)
#Trimming ctd data
k1 = np.argmin(np.abs(ctd_time - tstart))
k2 = np.argmin(np.abs(ctd_time - tend))

ctd_time = ctd_time[k1:k2+1]
sal = sal[k1:k2+1]
temp = temp[k1:k2+1]