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
plt.style.use('ggplot')


semaine = 26                   """ il n y a qu'un seul essai !! """
FB = "FB1"
ls = time(05, 36)                             
cs = time(21, 45)




#==============================================================================
"""---------------------------- data import ------------------------------- """ 
#==============================================================================

df0 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/26_2016_T3/FB1/ErrorLog_0_20160628-1007.csv",             
                  sep = ";", header = False , usecols=[0,23,24,27,28,29,30],
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']) 

df0 = df0[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]


df1 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/26_2016_T3/FB1/ErrorLog_20160704-103154.csv",             
                  sep = ";", header = False , usecols=[0,23,24,27,28,29,30],
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']) 

df1 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]




df2 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/26_2016_T3/FB1/Results_0_20160628-1007.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False,
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])  

#df3 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/26_2016_T3/FB1/Results_0_20160629-1704.csv",
#                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False,
#                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])                  
                                    

df4 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/26_2016_T3/FB1/Results_20160628-1007.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])  
                  
df5 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/26_2016_T3/FB1/ErrorLog_20160704-063056.csv",             
                  sep = ";", header = False , usecols=[0,23,24,27,28,29,30],
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']) 

df5 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]
                  
df6 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/26_2016_T3/FB1/ErrorLog_20160704-023004.csv",             
                  sep = ";", header = False , usecols=[0,23,24,27,28,29,30],
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']) 

df6 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]

df7 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/26_2016_T3/FB1/ErrorLog_20160703-222914.csv",             
                  sep = ";", header = False , usecols=[0,23,24,27,28,29,30],
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']) 
                  
df7 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]

df8 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/26_2016_T3/FB1/ErrorLog_20160703-182824.csv",             
                  sep = ";", header = False , usecols=[0,23,24,27,28,29,30],
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']) 
                  
df8 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]


df9 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/26_2016_T3/FB1/ErrorLog_20160703-182824.csv",             
                  sep = ";", header = False , usecols=[0,23,24,27,28,29,30],
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']) 
                  
df9 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]



df = pd.concat([df0, df1, df2, df4, df5, df6, df6, df7, df8, df8, df9] , axis=0, ignore_index=True)  
timestr = tm.strftime("%Y%m%d-%H%M%S")

nom_figure_moyenne = 'la courbe moyenne de la semaine {:}_2016_'.format(semaine)   +'{:}'.format(FB) +' {:}'.format(timestr) 
nom_figure_entropie = 'la courbe entropie de la semaine {:}_2016_'.format(semaine)  +'{:}'.format(FB) +' {:}'.format(timestr) 
file_name = 'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/obs_semaines/sem{:}_2016/'.format(semaine)

try:
    os.mkdir(file_name) 
except :
    pass

#dfr.to_csv('exemple_num_tetard_sem262016.csv', sep = ';')
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
    date_to_hour(df)   
    data26FB1 = df_semaine(df, file_name, semaine, FB)
    data26FB1_T3 = convert_to_hours(data26FB1)   
    del data26FB1
    #num_tetard(df)                               





#def entropie_plot(df, file_name, nom_figure_entropie):
#    dfr = df[['moyenne','ecart-type','entropie', 'num_essai','diff_date','num_tetard']]
#    dfr = dfr.reset_index(drop = True)
#    fig, ax = plt.subplots(1, 1, figsize =(18,10))
#    ax.plot(dfr[dfr["num_essai"] == 1].index, dfr[dfr["num_essai"] == 1].diff_date, 'o-', c = 'lawngreen', label ="diff entre deux obs") 
#    plt.legend()
#    plt.show()
#    return;