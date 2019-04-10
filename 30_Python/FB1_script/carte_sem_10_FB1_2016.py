# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:08:40 2016

@author: nabil.belahrach
"""

import pandas as pd
import matplotlib.pylab as plt
from WatchFrogpy import *
from datetime import datetime, timedelta, time
import time as tm
plt.style.use('ggplot')





semaine = 10
FB = "FB1"
ls = time(7,12)
cs = time(18,33)

#==============================================================================
"""---------------------------- data import ------------------------------- """ 
#==============================================================================

#df1 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/02_2016/FB1/ErrorLog_FB2_20160111-0826_2.csv",             
#                  sep = ";", header = False , usecols=[0,26,27,28,29,30,31],
#                  names = ['nom','solidite','entropie','smoothness','moyenne','ecart-type','mm_gradient']) 
#
#df1 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]



df2 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/10_2016/FB1/Results_20160308-1405.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])  

df3 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/10_2016/FB1/Results_20160309-0745.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])                  
                  # bon                  

df4 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/10_2016/FB1/Results_20160309-0745ET.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])    
     
df = pd.concat([df3, df4, df2] , axis=0, ignore_index=True)  
timestr = tm.strftime("%Y%m%d-%H%M%S")

nom_figure_moyenne = 'la courbe moyenne de la semaine 10_2016  FB1 {:}'.format(timestr)
nom_figure_entropie = 'la courbe entropie de la semaine 10_2016 FB1 {:}'.format(timestr)

file_name = 'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/obs_semaines/sem10_2016/'

#==============================================================================
# 
#==============================================================================
if __name__ == "__main__": 
    df = df_preparation(df)
    df = create_date_time_var(df)                                # contient tout ce qui précéde
    df = essai_nbr( df )     
    df = moyenne_par_essai(df)    
    moyenne_plot(df, file_name, nom_figure_moyenne)
    df = entropie_par_essai(df) 
    df = solidite_par_essai(df)
    df = sigma_par_essai(df)
    #df = entropie_par_essai(df)  
    #entropie_plot(df, file_name, nom_figure_entropie)
    date_to_hour(df)   
    data10 = df_semaine(df, file_name, semaine, FB)
    data10EVian = convert_to_hours(data10)
    #m10 = en_journee(df,ls, cs)