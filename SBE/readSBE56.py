#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:49:24 2018

@author: gegan
"""
import datetime
import glob
import numpy as np

#Year that data was collected (input)
year = 2018

#Path to the folder containing all of the .cnv files (input)
cnvpath = '/Users/gegan/Documents/Python/Research/SBE/CNV'

skip = len(cnvpath)
files = glob.glob(cnvpath+'/*.cnv')

SBE56data = dict()

"""This could probably be done in 1 line with pandas.read_csv, but I wrote this before
I used pandas"""

for ii in range(len(files)):
    name = files[ii]
    datetemp = np.array([],dtype='datetime64')
    temptemp = np.array([])
    with open(name,'r') as f:
        for _ in range(90):
            next(f)
        for line in f.readlines():
            ls = line.lstrip()
            _,jday,temp,_ = line.split()
            time = np.datetime64(datetime.datetime(year-1,12,31) + 
                                 datetime.timedelta(days=float(jday)))
            datetemp = np.append(datetemp,time)
            temptemp = np.append(temptemp,np.array(temp))
            SBE56data[name[skip+7:skip+12]] = ({'time':datetemp,
                   'temp':temptemp})

# Saving the dictionary
np.save('SBE56data.npy',SBE56data)
