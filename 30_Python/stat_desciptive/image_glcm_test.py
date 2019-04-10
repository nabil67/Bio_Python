# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:05:45 2016

@author: nabil.belahrach@veolia.com
"""

import numpy as np
import skimage.feature as sf
from skimage import io
from PIL import Image
import os


source = os.listdir("I:\\BIOTTOPE\\ImagesEvry\\Lecture Jour 3 dose r\xe9ponse semaine 44\\Evian")

#for element in source :
#    if element.endswith('.tif'):
#       #img=sf.greycomatrix(element, distances, angles, levels=256, symmetric=False, normed=False)
#       result = sf.greycomatrix(element, [2], [0], levels=256)
#       cont_val = sf.greycoprops(result,prop='contrast')
#       print("le contrast =", cont_val)
#          
#    #    print("'%s' est un fichier image" % element)
#    else:
#        pass
    

im = Image.open('I:\\BIOTTOPE\\ImagesEvry\\Lecture Jour 3 dose r\xe9ponse semaine 44\\Evian\\Snapshot-20141030-144809138.tif')
im.show()
img = np.array(im)
result = sf.greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],levels=256,  normed= True)
os.getcwd()