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


semaine = 14
FB = "FB2"

#==============================================================================
"""---------------------------- data import ------------------------------- """ 
#==============================================================================

#df1 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/14_2016/FB2/ErrorLog_20160405-1502.csv",             
#                  sep = ";", header = False , usecols=[0,23,24,27,28,29,30],
#                  names = ['nom','solidite','entropie','smoothness','moyenne','ecart-type','mm_gradient']) 
#
#df1 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]



df2 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/14_2016/FB2-opt/Results_20160405-1502.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])  
#
#df3 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/14_2016/FB2/Results_20160407-0749ET.csv",
#                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
#                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])                  
                                    

#df4 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/11_2016/FB1/Results_20160317-1643.csv",
#                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
#                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])    
    
df = pd.concat([ df2] , axis=0, ignore_index=True)  

strtime = tm.strftime("%Y-%m-%d-%H-%M")

nom_figure_moyenne = 'la courbe moyenne de la semaine {:}opt_2016'.format(semaine) +'ErrLog  FB2 {:}'.format(strtime)
nom_figure_entropie = 'la courbe entropie de la semaine {:}_2016'.format(semaine) +'ErrLog FB2  {:}'.format(strtime)

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
    #df = moyenne_norm(df)  
    moyenne_plot(df, file_name, nom_figure_moyenne)
    df = entropie_par_essai(df)  
    #entropie_plot(df, file_name, nom_figure_entropie)
    date_to_hour(df)     
    data14FB2opt = df_semaine(df, file_name, semaine, FB)
    data14FB2opt = convert_to_hours(data14FB2opt)                                  
