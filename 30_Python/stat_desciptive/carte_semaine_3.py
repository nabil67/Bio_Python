
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 14:59:51 2016

@author: nabil.belahrach
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib import dates
import seaborn as sns
from datetime import datetime
from datetime import timedelta 
from dateutil import tz
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
plt.style.use('ggplot')

#from_zone = tz.gettz('UTC')
#to_zone = tz.gettz('Europe/Paris')


#==============================================================================
"""----------------------------- Preparation de data ---------------------- """
#==============================================================================
def df_preparation():
    df0 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/ErrorLog_0.csv", 
                      sep = ";", header = False , usecols=[1,23,24,25,26,27,28,29], 
                      names = ['nom','maxg','moyenne','ecart-type','head_area','convex','solidity','entropy'])               

    df1 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/Results-18-21.csv", sep = ";",
                      header = False,usecols=[1,3,4,5,6,7,8,9],
                      names = ['nom','maxg','moyenne','ecart-type','head_area','convex','solidity','entropy'])
                      
    df0 = df0[ df0.entropy != "0"]                            # select wehere entropy !=0               
    df = pd.concat([df0,df1] , axis=0, ignore_index=True)     
    df['moyenne'] = df['moyenne'].str.replace("," , ".")
    df['ecart-type'] = df['ecart-type'].str.replace("," , ".")
    df['head_area'] = df['head_area'].str.replace("," , ".")
    df['convex'] = df['convex'].str.replace("," , ".")
    df['solidity'] = df['solidity'].str.replace("," , ".")
    df['entropy'] = df['entropy'].str.replace("," , ".")
    df['FrogBox'] = "1"
    df['OPT'] = "Non"
    df[['maxg','moyenne','ecart-type','head_area','convex','solidity','entropy']] = df[['maxg','moyenne','ecart-type','head_area','convex','solidity','entropy']].astype(float)      
                                        
    return df

#==============================================================================
"""----------------------- extraire la date et l'heure ---------------------"""
#==============================================================================
def date_heure_extract():   
    dmj = []
    hms = []
    for i in xrange(len(df)) :
            ll = ((df.nom[i].split("."))[0]).split("-")
            dmj.append(str(ll[1]))
            hms.append(str(ll[2]))
    return dmj, hms;
    

   
def create_date_time_var():
     dmj, hms = date_heure_create()
     date = []
     for i in xrange(len(df)):
         word1 = dmj[i]
         word2 = hms[i]
         year, month, day = int(word1[0:4]), int(word1[4:6]), int(word1[6:8])
         hour, mins, sec = int(word2[0:2]), int(word2[2:4]), int(word2[4:6]), 
         try :
             usec = int(word2[6:])*1000
         except ValuesError:
             usec = 0             
         otime =  datetime(year, month, day, hour, mins, sec, usec) 
         date.append(otime)  
     df["date"] = date                              #inclure la date dans df
     df[["date"]] = df[["date"]].astype(datetime)   
     df = df.sort(["date"])                         # ordoner par date
     df = df.reset_index(drop = True)               # actualiser l'Index    
     return date
     
def time_by_hours():
    date = create_date_time_var()
    time_by_hours = []
    base_date = df.date[0]
    for i in xrange(len(df)):        
        hours_diff = (df.date[i] - base_date).total_seconds() / 3600.0
        time_by_hours.append(hours_diff)
        i = i + 1        
    return time_by_hours
        
def plot_time_serie():
    values = pd.Series(df["moyenne"])
    ts = pd.Series(values, index = df.date)
    ts.plot(style='k--', label = 'La moyenne')
    plt.legend()
    ts.plot()
    fig, ax = plt.subplot()
    plt.plot(time_by_hours, df.moyenne)
    plt.ylabel('La moyenne')
    plt.legend()
    plt.grid()
    plt.show()
    return 
   
    
#==============================================================================
"""----------------------- la sortie_csv ---------------------------- """
#==============================================================================
 def serie_tompo_create():     
     df = df[['nom','date','FrogBox','OPT','maxg','moyenne','ecart-type','head_area','convex','solidity','entropy']]
     df.to_csv('U:/Stagiaires/Nabil.BELAHRACH/Donnees/20_brut/carte_semaine_3.csv', sep = ';')
     
#==============================================================================
"""------------------------ Manipulation des dates ------------------------ """
#==============================================================================



if __name__=='__main__':
    df_preparation()
    date_heure_extract()
    create_date_time_var()
    plot_time_serie()