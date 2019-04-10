# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:08:40 2016
@author: nabil.belahrach
"""

import pandas as pd
import matplotlib.pylab as plt
import time
from carte_sem_01_2016 import *
plt.style.use('ggplot')


semaine = 01
FB = "FB2" 

#==============================================================================
"""---------------------------- data import ------------------------------- """ 
#==============================================================================

df1 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/01_2016/FB2/ErrorLog_FB2_20160111-0826_2.csv",             
                  sep = ";", header = False , usecols=[0,26,27,28,29,30,31],
                  names = ['nom','solidite','entropie','smoothness','moyenne','ecart-type','mm_gradient']) 

df1 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]



df2 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/01_2016/FB2/Result_s01_FB2_20160105-1654.csv",
                  usecols=[1,3,4,7,8,9,10], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])  

df3 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/01_2016/FB2/Results_FB2_20160111-0826.csv",
                  usecols=[1,3,4,7,8,9,10], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])                  
                  # bon                  

df4 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/01_2016/FB2/Results_s01_BF2_20160107-0855.csv",
                  usecols=[1,3,4,7,8,9,10], sep = ";", header = False, 
                  names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])    
     
df = pd.concat([df3, df4 ,df1, df2] , axis=0, ignore_index=True)  

nom_figure_moyenne = 'la courbe moyenne de la semaine {:}_2016'.format(semaine) +'ErrLog  FB2 {:}'.format(time.strftime("%Y-%m-%d-%H:%M"))
nom_figure_entropie = 'la courbe entropie de la semaine {:}_2016'.format(semaine) +'ErrLog FB2  {:}'.format(time.strftime("%Y-%m-%d-%H:%M"))

file_name = 'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/obs_semaines/sem{:}_2016/'.format(semaine)


try:
    os.mkdir(file_name) 
except :
    pass


#==============================================================================
""""----------------------------- main() ---------------------------------- """
#==============================================================================

if __name__ == "__main__": 
    df = df_preparation(df)
    df = create_date_time_var(df)                                # contient tout ce qui précéde
    df = essaye_nbr(df)     
    df = moyenne_par_essaye(df)    
    df = moyenne_norm(df)  
    moyenne_plot(df, file_name, nom_figure_moyenne)
    df = entropie_par_essaye(df)  
    #entropie_plot(df, file_name, nom_figure_entropie)
    date_to_hour(df)   
    data01FB2 = df_semaine(df, file_name, semaine, FB)
    data01FB2 = convert_to_hours(data01FB2) 