# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:55:06 2016

@author: nabil.belahrach """
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from datetime import datetime, timedelta
import time
import seaborn as sns
from sklearn import preprocessing
from pandas.tools.plotting import scatter_matrix
plt.style.use('ggplot')
sns.set_style("whitegrid")

#from_zone = tz.gettz('UTC')
#to_zone = tz.gettz('Europe/Paris')
#from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
#import matplotlib.dates as mdates

df1 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/Results_18_FB2_20150428-1013.csv", sep = ";", header = False,)                   
                  names = ['Folder','nom','sequence','moyenne','ecart-type','size','pixel_max','fluorescent_pixel' ])
             

df2 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/Results_18_FB2_20150428-1228.csv", sep = ";", header = False,
                  names = ['Folder','nom','sequence','moyenne','ecart-type','size','pixel_max','fluorescent_pixel' ])                  

df3 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/Results_18_FB2_20150429-0831.csv", sep = ";", header = False,                              
                  names = ['Folder','nom','sequence','moyenne','ecart-type','size','pixel_max','fluorescent_pixel' ])                    

nom_figure_moyenne = 'la courbe moyenne de la semaine 18  FB 2'
nom_figure_maxpixel = 'la courbe pixel_max de la semaine 18 FB 2'

def df_preparation():    
    df = pd.concat([df1,df2, df3] , axis=0, ignore_index=True)  
    
    try :
        len(df) == len(df1) + len(df2) + len(df3)
    except False:
        print("Erreur dans la concaténation")
    df['moyenne'] = df['moyenne'].str.replace("," , ".")
    df['ecart-type'] = df['ecart-type'].str.replace("," , ".")
    df['size'] = df['size'].str.replace("," , ".")
    df['FrogBox'] = "1"
    df['OPT'] = "Non"
    df[['moyenne','ecart-type','size','pixel_max','fluorescent_pixel']] = df[['moyenne','ecart-type','size','pixel_max','fluorescent_pixel']].astype(float)                                              
    return df

#==============================================================================
"""----------------------- variable date ---------------------------------- """
#==============================================================================
def date_heure_extract(df):   
    df = df_preparation()
    dmj, hms = [], []
    for i in xrange(len(df)) :
            ll = ((df.nom[i].split("."))[0]).split("-")
            dmj.append(str(ll[1]))
            hms.append(str(ll[2]))
    return dmj, hms;
         
    
def create_date_time_var(df):
     df = df_preparation()
     dmj, hms = date_heure_extract(df)
     date = []
     for i in xrange(len(df)):
         word1 = dmj[i]
         word2 = hms[i]
         year, month, day = int(word1[0:4]), int(word1[4:6]), int(word1[6:8])
         hour, mins, sec = int(word2[0:2]), int(word2[2:4]), int(word2[4:6]), 
         try :
             usec = int(word2[6:])*1000
         except :
             usec = 0             
         otime =  datetime(year, month, day, hour, mins, sec, usec) 
         date.append(otime)
     df["date"] = date                              #inclure la date dans df 
     df[['date']] = df[['date']].astype(datetime)
     df = df.sort(["date"])                         # ordoner par date
     df = df.reset_index(drop = True) 
     df.index = df['date']                             # actualiser l'Index  
     df = df[['Folder','nom','sequence','FrogBox','OPT','moyenne','ecart-type','size','pixel_max','fluorescent_pixel']]
     return df;
     


#==============================================================================
"""--------------------------variable essaye ------------------------------ """     
#==============================================================================
def essaye_nbr(df):
    df = create_date_time_var(df)
    date = df.index
    num_essaye = [1]
    nbr = 1
    for i in  xrange(len(df) - 1):
        if ((date[i + 1 ] - date[i]) < timedelta(hours = 3)): 
            num_essaye.append(nbr)
        else:
            nbr = nbr + 1
            num_essaye.append(nbr)
    df["num_essaye"] =  num_essaye
    return df;
                                # bon !

    
def moyenne_par_essaye(df):
     m = (df[["moyenne", "num_essaye"]].groupby(['num_essaye']).mean()).values
     e = (df.num_essaye.value_counts(sort = False).sort_index()).values
     moyenne_par_essaye = []
     for i in xrange(len(e)): 
         tmp=np.repeat(m[i], e[i])
         moyenne_par_essaye.extend(tmp)
         i = i + 1
     pass
     df["moyenne_par_essaye"] = moyenne_par_essaye 
     return 
     
moyenne_par_essaye(df)                                 # bon !
 
def std_par_essaye(df):
     sigma_par_essaye = []
     sigma = (df[["moyenne", "num_essaye"]].groupby(['num_essaye']).std()).values
     e = (df.num_essaye.value_counts(sort = False).sort_index()).values
     for i in xrange(len(e)):
        tmp = np.repeat(sigma[i], e[i])
        sigma_par_essaye.extend(tmp)
        i = i + 1
     df["sigma_par_essaye"] = sigma_par_essaye 
     return ; 


#==============================================================================
"""------------------------normalisation de moyenne ----------------------- """
#==============================================================================

def moyenne_norm(df):
    tmp = df["moyenne"].values
    min_max_scaler = preprocessing.MinMaxScaler()
    moyenne_norm = min_max_scaler.fit_transform(tmp)
    df["moyenne_norm"] = moyenne_norm
    m = (df[["moyenne_norm", "num_essaye"]].groupby(['num_essaye']).mean()).values
    e = (df.num_essaye.value_counts(sort = False).sort_index()).values
    moyenne_norm_par_essaye = []
    for i in xrange(len(e)): 
        tmp=np.repeat(m[i], e[i])
        moyenne_norm_par_essaye.extend(tmp)
        i = i + 1
    pass
    df["moyenne_norm_par_essaye"] = moyenne_norm_par_essaye
    return ;
    

#==============================================================================
"""----------------------fluorescent_pixel ------------------------------------------"""
#==============================================================================
def pixel_max_par_essaye(df):
     m = (df[["pixel_max", "num_essaye"]].groupby(['num_essaye']).mean()).values
     e = (df.num_essaye.value_counts(sort = False).sort_index()).values
     pixel_max_par_essaye = []
     for i in xrange(len(e)): 
         tmp=np.repeat(m[i], e[i])
         pixel_max_par_essaye.extend(tmp)
         i = i + 1
     pass
     df["pixel_max_par_essaye"] = pixel_max_par_essaye
     return ;
    
    
#==============================================================================
"""--------------------------- sortie_csv ----------------------------------"""
#==============================================================================
# def csv_tompo_create():     
#     print("imprimer csv")
#     #df = df[['nom','date','FrogBox','OPT','num_essaye','maxg','moyenne','ecart-type','head_area','convex','solidity','entropy']]   
#     #df.to_csv('U:/Stagiaires/Nabil.BELAHRACH/Donnees/20_brut/Results_20_FB2.csv', sep = ';')
#     return
    
#==============================================================================
"""----------------------------- plot --------------------------------------"""
#==============================================================================

def moyenne_plot():
    fig1, ax1 = plt.subplots(1, 1, figsize =(18,10))
    ax1.plot(df.index, df.moyenne, 'o', c = 'olivedrab') 
    ax1.plot(df.index, df.moyenne_par_essaye, 'o-', c = 'royalblue', label = " la moyenne") 
    ax1.plot(df.index, df.moyenne_norm_par_essaye, 'o-', c = 'darkorange', label = " la moyenne normalisee") 
    datemin = df.index[1] - timedelta(hours = 2)
    datemax = df.index[len(df)-1] + timedelta(hours = 2)
    pass
    ax1.set_xlim(datemin, datemax)
    plt.title('nom_figure_moyenne')
    plt.xticks( rotation = 30)
    plt.legend()
    date_string = time.strftime("%Y-%m-%d-%H:%M")
    plt.savefig(nom_figure_moyenne)
    plt.show()
    return;


def pixel_max_plot():
    fig, ax = plt.subplots(1, 1, figsize =(18,10))
    datemin = df.index[1] - timedelta(hours = 2)
    datemax = df.index[len(df)-1] + timedelta(hours = 2)
    ax.set_xlim(datemin, datemax)
    ax.plot(df.index, df.pixel_max, 'o', c = 'olivedrab') 
    ax.plot(df.index, df.pixel_max_par_essaye, 'o-', c = 'royalblue', label = " moyenne pixel_max_par_essaye") 
    plt.axhline( df.pixel_max.mean()  , c='k', label='moyenne pixel_max ', linestyle='-', linewidth=4)
    pass
    plt.title(nom_figure_maxpixel)
    plt.xticks( rotation = 30)
    plt.legend()
    date_string = time.strftime("%Y-%m-%d-%H:%M")
    plt.savefig(nom_figure_maxpixel)
    plt.show()
    return;


    
    df.boxplot(column="moyenne", by="num_essaye")   
    
    

#==============================================================================
"""----------------------------- main() ----------------------------------- """
#==============================================================================



if __name__=='__main__':
    df = df_preparation(); 
    df = essaye_nbr(df) 
    df = moyenne_par_essaye(df)  
    pixel_max_par_essaye(df)
    moyenne_plot()
    pixel_max_plot() 






#    data  = df[["moyenne","ecart-type","size","fluorescent_pixel","pixel_max"]]
#    scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
#    plt.savefig('matrice_corr_S18')


#    ax1.plot(df.index, df.moyenne_par_essaye + 2* df.sigma_par_essaye, 'o--', c='g', label ="LS") 
#    ax1.plot(df.index, df.moyenne_par_essaye - 2* df.sigma_par_essaye, 'o--', c='g') 
#    
#    sns.violinplot(y=df["moyenne"])
#    ax1.boxplot(df["moyenne"], df.index)
#    ax = sns.violinplot("num_essaye", "moyenne", data = df, jitter = True)    
#    plot_time_serie()   
#ax.xaxis.set_major_locator(DayLocator())
#ax.xaxis.set_minor_locator(HourLocator(np.arange(0, 25, 1)))
#ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
#ax.xaxis.grid(True, 'minor')
#ax.xaxis.grid(True, 'major')
   
#    plt.axhline( df.moyenne.mean() + 3*df.moyenne.std() , c='r', label='LCS', linestyle='-')
#    plt.axhline( df.moyenne.mean() - 3*df.moyenne.std() , c='r', label='LCI', linestyle='-')
#    plt.axhline( df.moyenne.mean() + 2*df.moyenne.std() , c='g', label='LSS', linestyle='--')
#    plt.axhline( df.moyenne.mean() - 2*df.moyenne.std() , c='g', label='LSI', linestyle='--')
   
   
#def rolling_mean_plot():
#    fig2, ax = plt.subplots(1, 1, figsize =(16,10))
#    datemin = df.index[1] - timedelta(hours = 1)
#    datemax = df.index[len(df)-1] + timedelta(hours = 1)
#    pass
#    ax.set_xlim(datemin, datemax)
#    curve_200 = pd.rolling_mean(df["moyenne"] , 200, min_periods = df.moyenne[0])
#    curve_300 = pd.rolling_mean(df["moyenne"] , 300, min_periods = df.moyenne[0])  
#    curve_500 = pd.rolling_mean(df["moyenne"] , 500, min_periods = df.moyenne[0])  
#    curve_200.plot(label ='curve 200', legend= True)
#    curve_300.plot(label ='curve_300', legend= True)
#    curve_500.plot(label ='curve_5000', legend= True)
#    plt.title('rolling_mean semaine 20 FB 1')
#    
#    roll_mean = pd.ewma(df.moyenne, 500)
#    plt.plot(roll_mean.index, roll_mean , 'o')
##    plt.savefig('rolling_mean_sem18_FB_1.png')        
#    return