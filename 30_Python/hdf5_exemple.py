# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:00:08 2016

@author: nabil.belahrach
"""

import numpy as np
import h5py
import pylab

x = np.linspace(0,4*np.pi,1000)
y = np.sin(x) + 0.2*np.sin(10*x + 2.5*np.sin(7*x))

f = h5py.File('foo.h5', 'w')
f['axis'] = x
f['data'] = y
f.close()