# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:08:40 2016

@author: nabil.belahrach
"""

import pandas as pd
import matplotlib.pylab as plt
from WatchFrogpy import *
import time as tm
import os
import scipy as cp
plt.style.use('ggplot')


semaine = 28
FB = "FB1"
ls = time(05, 36)
cs = time(21, 45)

#==============================================================================
"""---------------------------- data import ------------------------------- """ 
#==============================================================================

df1 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/28_2016/FB1/ErrorLog0.csv",             
                  sep = ";", header = False , usecols=[0,28,24,27,28,29,30],
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']) 
#
df1 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]



df2 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/28_2016/FB1/Results_20160607-1132.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])  

df3 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/28_2016/FB1/Results0.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])                  
                                    

#df4 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/28_2016/FB2/Results0_20160528-0851.csv",
#                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
#                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])    
    
df = pd.concat([df3, df2, df1] , axis=0, ignore_index=True)  
timestr = tm.strftime("%Y%m%d-%H%M%S")

nom_figure_moyenne = 'la courbe moyenne de la semaine 28_2016   {:}'.format(FB) +' {:}'.format(timestr) 
nom_figure_entropie = 'la courbe entropie de la semaine 28_2016   {:}'.format(FB) +' {:}'.format(timestr) 
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
    data28 = df_semaine(df, file_name, semaine, FB)
    data28 = convert_to_hours(data28)  
    m28 = en_journee(df,ls, cs)                                
#==============================================================================
"""----------------------------test de student------------------------------"""
#==============================================================================

#def ttest_jour_nuit():        
#    m01_28 = pd.concat([m01, m10, m11, m14, m21, m22, m28] , axis=0, ignore_index=True)  
#    semaine = [01,10,11,14,21,22,28]
#    m01_28["semaine"] = semaine
#    m01_28 = m01_28[["semaine", jour, nuit]]
#    m01_28.columns.rename(1, "jour")
#    m01_28.columns.rename(0, "nuit")
#    d = {'semaine': m01_28.semaine, 'nuit': m01_28[0], 'jour' : m01_28[1]}
#    X = m01_28.values
#    cp.stats.ttest_ind(m01_28["jour"], m01_28["nuit"], equal_var = False)
#    m01_28.to_csv( 'm01_28.csv',sep =';', header = True)
#    m01_28[["jour", "nuit"]].hist( groupeby= semaine)
#    return;