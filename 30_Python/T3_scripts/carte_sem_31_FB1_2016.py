# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:08:40 2015

@author: nabil.belahrach
"""

import pandas as pd
import matplotlib.pylab as plt
from WatchFrogpy import *
import time as tm
import os
plt.style.use('ggplot')

semaine = "31_2016"
FB = "FB1"

#==============================================================================
"""---------------------------- data import ------------------------------- """ 
#==============================================================================

df1 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/31_2016_T3/FB1/ErrorLog.csv",             
                  sep = ";", header = False , usecols=[0,23,24,27,28,29,30],
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']) 


df1 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]



df2 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/31_2016_T3/FB1/Results_0_20160802-1506.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False,
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])  

df3 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/31_2016_T3/FB1/Results_20160802-1506.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False,
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])                  
                                    
#
#df4 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/41_2015_T3/FB1/Results_20151019-1006.csv",
#                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
#                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])    
#                  
#df5 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/41_2015_T3/FB1/Results_20151021-0859.csv",
#                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
#                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']) 
    
df = pd.concat([df1, df2, df3] , axis=0, ignore_index=True)  

timestr = tm.strftime("%Y%m%d-%H%M%S")
nom_figure_moyenne = 'la courbe moyenne T3 2,25 de la semaine {:}_'.format(semaine)   +'{:}'.format(FB) +' {:}'.format(timestr) 
nom_figure_entropie = 'la courbe entropie T3 2,25 de la semaine {:}_'.format(semaine)  +'{:}'.format(FB) +' {:}'.format(timestr) 
file_name = 'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/obs_semaines/sem{:}/'.format(semaine)

try:
    os.mkdir(file_name) 
except :
    pass

#==============================================================================
"""----------------- main() ----------------------------------------------- """
#==============================================================================
if __name__ == "__main__": 
    df = df_preparation(df)
    df = create_date_time_var(df)                                # contient tout ce qui précéde
    df = essai_nbr(df)     
    df = moyenne_par_essai(df)    
    moyenne_plot(df, file_name, nom_figure_moyenne)
    df = entropie_par_essai(df) 
    df = solidite_par_essai(df)
    df = sigma_par_essai(df)
    entropie_plot(df, file_name, nom_figure_entropie)
    date_to_hour(df)   
    data31FB1_2016_T3 = df_semaine(df, file_name, semaine, FB)
    data31FB1_2016_T3 = convert_to_hours(data31FB1_2016_T3)      
    data31FB1_2016_T3["semaine_et_FB"] = "data31FB1_2016_T3"       
#    del data41_2015FB1                     
    #num_tetard(df)
    
    