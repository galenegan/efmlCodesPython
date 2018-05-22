#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 18:22:46 2018

@author: gegan
"""
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pyrsktools as prsk


rsk = prsk.open('sample.rsk')

data = rsk.npsamples()

time = data['timestamp']
pressure = data['pressure']
temp = data['temperature']

plt.figure(1)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gcf().autofmt_xdate()

plt.plot(time,temp)
plt.xlabel('Time')
plt.ylabel('Temp')