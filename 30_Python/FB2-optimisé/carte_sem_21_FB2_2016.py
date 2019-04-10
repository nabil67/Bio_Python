# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:08:40 2016

@author: nabil.belahrach
"""

import pandas as pd
import matplotlib.pylab as plt
from WatchFrogpy import *
import os
import time as tm

plt.style.use('ggplot')


semaine = 21
FB = "FB2"

#==============================================================================
"""---------------------------- data import ------------------------------- """ 
#==============================================================================

#df1 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/21_2016/FB2/ErrorLog_20160523-0851.csv",             
#                  sep = ";", header = False , usecols=[0,23,24,27,28,29,30],
#                  names = ['nom','solidite','entropie','smoothness','moyenne','ecart-type','mm_gradient']) 
#
#df1 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]



df2 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/21_2016/FB2-opt/Results_20160520-1626.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])  

df3 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/21_2016/FB2-opt/Results_20160523-0851.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])                  
                                    

df4 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/21_2016/FB2-opt/Results_20160602-0810.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])    
                  
df = pd.concat([ df2, df3, df4] , axis=0, ignore_index=True)  
    
strtime = tm.strftime("%Y-%m-%d-%H-%M")

nom_figure_moyenne = 'la courbe moyenne de la semaine {:}opt_2016'.format(semaine) +'  FB2 {:}'.format(strtime)
nom_figure_entropie = 'la courbe entropie de la semaine {:}_2016'.format(semaine) +' FB2  {:}'.format(strtime)

file_name = 'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/obs_semaines/sem{:}_2016/'.format(semaine)

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
    moyenne_plot(df, file_name, nom_figure_moyenne)
    df = entropie_par_essai(df)  
    #entropie_plot(df, file_name, nom_figure_entropie)
    date_to_hour(df)   
    data21FB2opt = df_semaine(df, file_name, semaine, FB)
    data21FB2opt = convert_to_hours(data21FB2opt)                                  
