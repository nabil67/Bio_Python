# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:08:40 2016

@author: nabil.belahrach
"""

import pandas as pd
import matplotlib.pylab as plt
from carte_sem_01_2016 import *
import os
plt.style.use('ggplot')


semaine = 12
FB = "FB1" 

#==============================================================================
"""---------------------------- data import ------------------------------- """ 
#==============================================================================
#df1 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/02_2016/FB1/ErrorLog_FB2_20160111-0826_2.csv",             
#                  sep = ";", header = False , usecols=[0,26,27,28,29,30,31],
#                  names = ['nom','solidite','entropie','smoothness','moyenne','ecart-type','mm_gradient']) 
#
#df1 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]



df2 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/12_2016/FB1/Results_20160324-1159.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])  

df3 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/12_2016/FB1/Results_20160325-1003.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])                  
                  # bon                  

#df4 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/11_2016/FB1/Results_20160317-1643.csv",
#                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
#                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])    
  
df = pd.concat([df3, df2] , axis=0, ignore_index=True)  


nom_figure_moyenne = 'la courbe moyenne de la semaine 12_2016  FB1 {:}'.format(time.strftime("%Y-%m-%d-%H:%M"))
nom_figure_entropie = 'la courbe entropie de la semaine 12_2016 FB1 {:}'.format(time.strftime("%Y-%m-%d-%H:%M"))

file_name = 'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/obs_semaines/sem12_2016/'
try:
    os.mkdir(file_name) 
except :
    pass

def convert_to_hours(ts):
    ts = data12
    delta_to_hours = []
    for i in xrange(len(ts)):
        delta_to_hours.append(ts.hours_debut_essai[i].total_seconds() / float(3600) )     #to_hours
    ts["delta_to_hours"] = delta_to_hours
    return ts;
  
#==============================================================================
# 
#==============================================================================
if __name__ == "__main__": 
    df = df_preparation(df)
    df = create_date_time_var(df)                                # contient tout ce qui précéde
    df = essaye_nbr( df )     
    df = moyenne_par_essai(df)    
    #df = moyenne_norm(df)  
#    moyenne_plot(df, file_name, nom_figure_moyenne)
    df = entropie_par_essai(df)  
#    entropie_plot(df, file_name, nom_figure_entropie)
    date_to_hour(df)   
    data12=df_semaine(df, file_name, semaine, FB)
    data12 = convert_to_hours(data12) 
