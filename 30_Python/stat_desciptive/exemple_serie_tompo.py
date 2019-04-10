# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:47:03 2016

@author: nabil.belahrach
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates
import datetime

a = np.array([
    [1293605162197, 500, 1000],
    [1293605477994, 200, 300],
    [1293605478057, 50, 150],
    [1293605478072, 2500, 2000],
    [1293606162213, 500, 1000],
    [1293606162229, 200, 600]])

d = a[:,0]
y1 = a[:,1]
y2 = a[:,2]

# convert epoch to matplotlib float format
s = d/1000
ms = d-1000*s  # not needed?
dts = map(datetime.datetime.fromtimestamp, s)
fds = dates.date2num(dts) # converted

# matplotlib date format object
hfmt = dates.DateFormatter('%m/%d %H:%M')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.vlines(fds, y2, y1)

ax.xaxis.set_major_locator(dates.MinuteLocator())
ax.xaxis.set_major_formatter(hfmt)
ax.set_ylim(bottom = 0)
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=.3)

plt.savefig('SerieTemporelle_VerticalLabel.png')
plt.show()