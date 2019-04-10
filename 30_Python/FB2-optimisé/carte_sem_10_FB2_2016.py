# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:08:40 2016

@author: nabil.belahrach
"""

import pandas as pd
import matplotlib.pylab as plt
from WatchFrogpy import *
import time as tm
plt.style.use('ggplot')



semaine = 10
FB = "FB2"
#==============================================================================
"""---------------------------- data import ------------------------------- """ 
#==============================================================================

#df1 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/02_2016/FB1/ErrorLog_FB2_20160111-0826_2.csv",             
#                  sep = ";", header = False , usecols=[0,26,27,28,29,30,31],
#                  names = ['nom','solidite','entropie','smoothness','moyenne','ecart-type','mm_gradient']) 
#
#df1 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]



#df2 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/10_2016/FB2/Results_20160308-1405.csv",
#                  usecols=[1,3,4,7,8,9,10], sep = ";", header = False, 
#                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])  

df3 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/10_2016/FB2-opt/Results_20160308-1410.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])                  
                  # bon                  

df4 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/10_2016/FB2-opt/Results_20160308-1439.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])    
                  

df2 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/10_2016/FB2-opt/Results_20160310-0800.csv",
                  usecols=[1,4,5,8,9,10,11], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']) 
     
df = pd.concat([df2, df3, df4] , axis=0, ignore_index=True)  

strtime = tm.strftime("%Y-%m-%d-%H-%M")
nom_figure_moyenne = 'la courbe moyenne de la semaine {:}opt_2016'.format(semaine) +'ErrLog  FB2 {:}'.format(strtime)
nom_figure_entropie = 'la courbe entropie de la semaine {:}_2016'.format(semaine) +'ErrLog FB2  {:}'.format(strtime)


file_name = 'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/obs_semaines/sem10_2016/'


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
    data10FB2opt = df_semaine(df, file_name, semaine, FB)
    data10FB2opt = convert_to_hours(data10FB2opt)   