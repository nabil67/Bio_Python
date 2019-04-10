# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:08:40 2015

@author: nabil.belahrach
"""

import pandas as pd
import matplotlib.pylab as plt
from WatchFrogpy import *
from datetime import datetime, timedelta, time
import time as tm
import os
import scipy as cp
plt.style.use('ggplot')
import sqlite3


semaine = 45
FB = "FB1"
ls = time(05, 36)
cs = time(21, 45)

#==============================================================================
"""---------------------------- data import ------------------------------- """ 
#==============================================================================

#df1 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/45_2015/FB1/ErrorLog0.csv",             
#                  sep = ";", header = False , usecols=[0,23,24,27,28,29,30],
#                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']) 
##
#df1 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]



df2 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/45_2015/FB1/Results_20151103-1034.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])  

df3 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/45_2015/FB1/Results_20151106-0932.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])                  
                                    

#df4 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/45_2015_T3/FB1/Results_20151125-0902.csv",
#                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
#                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])    
    
df = pd.concat([df3, df2] , axis=0, ignore_index=True)  
timestr = tm.strftime("%Y%m%d-%H%M%S")

nom_figure_moyenne = 'la courbe moyenne de la semaine 45_2015   {:}'.format(FB) +' {:}'.format(timestr) 
nom_figure_entropie = 'la courbe entropie de la semaine 45_2015   {:}'.format(FB) +' {:}'.format(timestr) 
file_name = 'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/obs_semaines/sem{:}_2015/'.format(semaine)

try:
    os.mkdir(file_name) 
except :
    pass

#==============================================================================
# 
#==============================================================================
if __name__ == "__main__": 
    df = df_preparation(df)
    df = create_date_time_var(df)                                # contient tout ce qui précéde
    df = essai_nbr(df)   
    df = moyenne_par_essai(df)    
    #df = moyenne_norm(df)  
    moyenne_plot(df, file_name, nom_figure_moyenne)
    df = entropie_par_essai(df)  
    #entropie_plot(df, file_name, nom_figure_entropie)
    date_to_hour(df)   
    #$data45 = df_semaine(df, file_name, semaine, FB)
    #data45 = convert_to_hours(data45)  
    #m41 = en_journee(df,ls, cs)        
    pass
    del df["date"] 
    df45_FB1_2015 = df
                     



