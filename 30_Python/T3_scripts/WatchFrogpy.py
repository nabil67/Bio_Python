# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:55:06 2016

@author: nabil.belahrach """
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from datetime import datetime, timedelta, time
#import time
from sklearn import preprocessing
plt.style.use('ggplot')

global semaine , FB, file_name, df

def data_import(semaine, FB):
    df1 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/01_2016/FB1/ErrorLog_01_FB1_20160106-0824.csv",             
                     sep = ";", header = False , usecols=[0,26,27,28,29,30,31],
                     names = ['nom','solidite','entropie','smoothness','moyenne','ecart-type','mm_gradient']) 
 
    df1 = df1[['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient']]



    df2 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/01_2016/FB1/Results_01_FB1_20160106-0824.csv",
                     usecols=[1,3,4,7,8,9,10], sep = ";", header = False, 
                     names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])                  
                  # bon                  
 
    df3 = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/fichiers_csv/01_2016/FB1/Results_T0_01_FB1_20160106-0824.csv",
                     usecols=[1,3,4,7,8,9,10], sep = ";", header = False, 
                     names = ['nom','moyenne','ecart-type','solidite','entropie','smoothness','mm_gradient'])    
     
    df = pd.concat([df3,df1, df2] , axis=0, ignore_index=True)  
    return;

def df_preparation(df):   
    df['moyenne'] = df['moyenne'].str.replace("," , ".")
    df['ecart-type'] = df['ecart-type'].str.replace("," , ".")
    df['solidite'] = df['solidite'].str.replace("," , ".")
    df['entropie'] = df['entropie'].str.replace("," , ".")
    df['smoothness'] = df['smoothness'].str.replace("," , ".")
    df['mm_gradient'] = df['mm_gradient'].str.replace("," , ".")
    df = df[df['moyenne'] != ' ']                                            
    df[['moyenne']] = df[['moyenne']].astype(float)
    df[['ecart-type']] = df[['ecart-type']].astype(float)  
    df[['solidite']] = df[['solidite']].astype(float)  
    df[['entropie']] = df[['entropie']].astype(float)   
    df[['smoothness']] = df[['smoothness']].astype(float)
    df[['mm_gradient']] = df[['mm_gradient']].astype(float)                                              
    df = df[df['moyenne'] != 0 ] 
    return df;
    

#==============================================================================
"""----------------------- variable date ---------------------------------- """
#==============================================================================
def date_heure_extract(df):   
    df = df.reset_index(drop = True)
    dmj, hms = [], []
    for i in xrange(len(df)):
            ll = ((df.nom[i].split("."))[0]).split("-")
            dmj.append(str(ll[1]))
            hms.append(str(ll[2]))
    return dmj, hms;
         
    
def create_date_time_var(df):
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
     df["date"] = date                                 #inclure la date dans df 
     df[['date']] = df[['date']].astype(datetime)
     df = df.sort(["date"])                            # ordoner par date
     df = df.reset_index(drop = True) 
     df.index = df['date']                            # actualiser l'Index  
     return df;
     


#==============================================================================
"""--------------------------variable essai ------------------------------ """     
#==============================================================================

def essai_nbr(df):
    date = df.index
    num_essai = [1]
    nbr = 1
    for i in  xrange(len(df) - 1):
        if ((date[i + 1 ] - date[i]) < timedelta(hours = 3)): 
            num_essai.append(nbr)
        else:
            nbr = nbr + 1
            num_essai.append(nbr)
    df["num_essai"] =  num_essai
    return df;


def moyenne_par_essai(df):
     m = (df[["moyenne", "num_essai"]].groupby(['num_essai']).mean()).values
     e = (df.num_essai.value_counts(sort = False).sort_index()).values
     moyenne_par_essai = []
     for i in xrange(len(e)): 
         tmp=np.repeat(m[i], e[i])
         moyenne_par_essai.extend(tmp)
         i = i + 1
     pass
     df["moyenne_par_essai"] = moyenne_par_essai 
     return df;

                              # bon !
 
def sigma_par_essai(df):
     m = (df[["ecart-type", "num_essai"]].groupby(['num_essai']).mean()).values
     e = (df.num_essai.value_counts(sort = False).sort_index()).values
     sigma_par_essai = []
     for i in xrange(len(e)):
        tmp = np.repeat(m[i], e[i])
        sigma_par_essai.extend(tmp)
        i = i + 1
     df["sigma_par_essai"] = sigma_par_essai 
     return df; 



def entropie_par_essai(df):
     m = (df[["entropie", "num_essai"]].groupby(['num_essai']).mean()).values
     e = (df.num_essai.value_counts(sort = False).sort_index()).values
     entropie_par_essai = []
     for i in xrange(len(e)): 
         tmp=np.repeat(m[i], e[i])
         entropie_par_essai.extend(tmp)
         i = i + 1
     pass
     df["entropie_par_essai"] = entropie_par_essai
     return df; 

def solidite_par_essai(df):
     m = (df[["solidite", "num_essai"]].groupby(['num_essai']).mean()).values
     e = (df.num_essai.value_counts(sort = False).sort_index()).values
     solidite_par_essai = []
     for i in xrange(len(e)): 
         tmp=np.repeat(m[i], e[i])
         solidite_par_essai.extend(tmp)
         i = i + 1
     pass
     df["solidite_par_essai"] = solidite_par_essai
     return df; 
     


#==============================================================================
"""----------------------------- plot --------------------------------------"""
#==============================================================================
def moyenne_plot(df, file_name, nom_figure_moyenne):
    fig1, ax1 = plt.subplots(1, 1, figsize =(18,10))
    ax1.plot(df.index, df.moyenne, 'o', c = 'olivedrab', label ="Passage des tetards") 
    ax1.plot(df.index, df.moyenne_par_essai, 'o-', c = 'royalblue', label = " la moyenne") 
    #ax1.plot(df.index, 30 * df.moyenne_norm_par_essai, 'o-', c = 'darkorange', label = " la moyenne 30 * normalisee") 
    datemin = df.index[1] - timedelta(hours = 2)
    datemax = df.index[len(df)-1] + timedelta(hours = 2)
    pass
    ax1.set_xlim(datemin, datemax)
    plt.title(nom_figure_moyenne)
    plt.xticks( rotation = 30)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Moyenne Fluo")
    plt.savefig(file_name + nom_figure_moyenne +'.png')
    plt.show()
    return;


def entropie_plot(df, file_name, nom_figure_entropie):
    fig, ax = plt.subplots(1, 1, figsize =(18,10))
    datemin = df.index[1] - timedelta(hours = 3)
    datemax = df.index[len(df)-1] + timedelta(hours = 3)
    ax.set_xlim(datemin, datemax)
    ax.set_ylim(df.entropie.min() - 0.1, df.entropie.max() + 0.1)
    ax.plot(df.index, df.entropie, 'o', c = 'olivedrab', label ="passages des tetard") 
    ax.plot(df.index, df.entropie_par_essai, 'o-', c = 'royalblue', label = " la courbe moyenne:  entropie par essai")
    ax.axhline(df.entropie.mean(),  c = 'darkorange', label = "moyenne d'entropie", alpha=1) 
    pass
    plt.title(nom_figure_entropie)
    plt.xticks( rotation = 30)
    plt.legend()
    plt .savefig(file_name + nom_figure_entropie +'.png')
    plt.show()
    return;
#==============================================================================
"""-------------------------  changement derepère temporelle----------------"""
#==============================================================================
def date_to_hour(df):                           
    date_to_hour = [timedelta(hours = 0)]
    for i in xrange(len(df) - 1):
        diff = (df.date[i+1] - df.date[0])
        date_to_hour.append(diff)        
    df["date_to_hour"]= date_to_hour
    return;   
#==============================================================================
"""-------------- sortie d = { 'hours_debut': datetime ,'mean_essai '}------"""
#==============================================================================    
def df_semaine(df, file_name, semaine, FB):
    hours_debut_essai = [timedelta(hours = 0)]
    mean_essai_sem = [df.moyenne_par_essai[0]]
    entropie_essai_sem = [df.entropie_par_essai[0]]
    solidite_essai_sem = [df.solidite_par_essai[0]]
    sigma_essai_sem = [df.sigma_par_essai[0]]
    for i in xrange( len(df) - 1):
        if ((df.date[i + 1 ] - df.date[i]) > timedelta(hours = 3)):
            hours_debut_essai.append(df.date_to_hour[i])
            mean_essai_sem.append(df.moyenne_par_essai[i])
            entropie_essai_sem.append(df.entropie_par_essai[i])
            solidite_essai_sem.append(df.solidite_par_essai[i])
            sigma_essai_sem.append(df.sigma_par_essai[i])
        else:
            pass
    d = {'hours_debut_essai': hours_debut_essai, 'mean_essai_sem': mean_essai_sem, 'entropie_essai_sem': entropie_essai_sem,
         'solidite_essai_sem': solidite_essai_sem, 'sigma_essai_sem': sigma_essai_sem }
    sem = pd.DataFrame( data = d)
    return sem;   
#==============================================================================
"""-------------- sortie d = { 'hours_debut': heure ,'mean_essai '}---------"""
#==============================================================================   
def convert_to_hours(ts):
    delta_to_hours = []
    for i in xrange(len(ts)):
        delta_to_hours.append(ts.hours_debut_essai[i].total_seconds() / float(3600) )     #to_hours
    ts["delta_to_hours"] = delta_to_hours
    return ts;

#==============================================================================
"""------------------- Séparer le jour et la nuit --------------------------"""
#==============================================================================
def en_journee(df,ls, cs):             # ls : levée de soleil
    jour = []                          #cs : couché de soleil
    for i in xrange(len(df)):
        if (ls <= time(df.date[i].hour, df.date[i].minute) <=  cs) :
            jour.append(1)
        else:
            jour.append(0)
    pass
    df["jour"] = jour
    m = (df[["moyenne", "jour"]].groupby(['jour']).mean()).T
    return m;

#==============================================================================
"""------------------- numeroter les tetatrds ------------------------------"""
#==============================================================================
def num_tetard(df):
     h = timedelta(seconds = 1./23).total_seconds()    #pas de temps <-- 30 images par seconde
     num_tetard = [1]
     nbr = 1
     v = [timedelta(0,0,0,0,0,0)]
     for i in xrange(1,len(df)): 
         diff = df.index[i] - df.index[i-1]
         v.append(diff.total_seconds())        
         if (diff.total_seconds() <= h):
             num_tetard.append(nbr)
         else:  
             nbr = nbr + 1   
             num_tetard.append(nbr)  
     df["diff_date"] = v
     df["num_tetard"] = num_tetard
     return df;     

#==============================================================================
"""----------------------------- main() ----------------------------------- """
#==============================================================================
if __name__=='__main__':    
    df = df_preparation(df)
    create_date_time_var(df)                                # contient tout ce qui précéde
    essai_nbr(df)     
    moyenne_par_essai(df)     
    moyenne_plot()
    entropie_par_essai(df)  
    solidite_par_essai(df)
    sigma_par_essai(df)
    entropie_plot()
    date_to_hour(df)   
    df_semaine(df, file_name, semaine, FB)
    convert_to_hours(ts) 
    en_journee(ls, cs)
    num_tetard(df)
    



#==============================================================================
"""--------------------------- sortie_csv ----------------------------------"""
#==============================================================================
# def csv_tompo_create():     
#     print("imprimer csv")
#     #df = df[['nom','date','FrogBox','OPT','num_essai','maxg','moyenne','ecart-type','head_area','convex','solidity','entropy']]   
#     df.to_csv('exemple_donnees.csv', sep = ';')
#     return   
#    data  = df[["moyenne","ecart-type","size","fluorescent_pixel","pixel_max"]]
#    scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
#    plt.savefig('matrice_corr_S18')


#    ax1.plot(df.index, df.moyenne_par_essai + 2* df.sigma_par_essai, 'o--', c='g', label ="LS") 
#    ax1.plot(df.index, df.moyenne_par_essai - 2* df.sigma_par_essai, 'o--', c='g') 
#    
#    sns.violinplot(y=df["moyenne"])
#    ax1.boxplot(df["moyenne"], df.index)
#    ax = sns.violinplot("num_essai", "moyenne", data = df, jitter = True)    
#    plot_time_serie()   
#    ax.xaxis.set_major_locator(DayLocator())
#    ax.xaxis.set_minor_locator(HourLocator(np.arange(0, 25, 1)))
#    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
#    ax.xaxis.grid(True, 'minor')
#    ax.xaxis.grid(True, 'major')
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