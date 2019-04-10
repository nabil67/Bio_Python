#-*- coding: utf-8 -*-
"""
Created on Thu Jun 02 16:52:49 2016
@author: nabil.belahrach
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import time
plt.style.use('ggplot')

#global data01FB2, data10FB2, data11FB2, data14FB2, data21FB2, data22FB2;
datas = [data13FB2, data26FB2, data27FB2]
FB = "FB2"
strtime = tm.strftime("%Y-%m-%d-%H:%M")
file_name = 'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/obs_semaines/T3_moyenne_TTsemaines'
nom_figure_moyenne = 'la courbe moyenne de 2016 {:}'.format(FB) +'{:}'.format(strtime)

#def data_rename_essais():
#    #for dfm in [data01FB2, data10FB2, data11FB2, data14FB2, data21FB2, data22FB2]:
#        data22FB2 = data22FB2.rename(columns = {'mean_essaye_sem':'mean_essai_sem'})       # Rename
#        data21FB2 = data21FB2.rename(columns = {'mean_essaye_sem':'mean_essai_sem'})
#        data01FB2 = data01FB2.rename(columns = {'mean_essaye_sem':'mean_essai_sem'})
#        data10FB2 = data10FB2.rename(columns = {'mean_essaye_sem':'mean_essai_sem'})
#        data11FB2 = data11FB2.rename(columns = {'mean_essaye_sem':'mean_essai_sem'})
#        data14FB2 = data14FB2.rename(columns = {'mean_essaye_sem':'mean_essai_sem'})
#        return;
#==============================================================================
"""------- normalisation par 1 ere valeur && index = heure de debut----------- """    
#==============================================================================
def moyenne_norm():
    for data in [data13_2016FB1, data41_2015FB1, data43_2015FB1, data46_2015FB1]:        
        data.index = data.delta_to_hours
        moyenne_norm_par_essai = data.mean_essai_sem / data.mean_essai_sem[0]
        data["moyenne_norm_par_essai"] = moyenne_norm_par_essai
    return;   
moyenne_norm()

data = pd.concat([ data13FB1] , axis=0, ignore_index=True)
data = data13FB1
#==============================================================================
"""---------------- graphique des semaines -------------------------------- """
#==============================================================================
def semaine_plot():
    fig1, ax1 = plt.subplots(1, 1, figsize =(18,10))
    #ax1.plot(data13FB2.index, data13FB2.mean_essai_sem, 'o-', c = 'r', label = " moyenne des essais semaine 13")
    ax1.plot(data26FB2.index, data26FB2.mean_essai_sem, 'o-', c = 'blue', label = " moyenne des essais semaine 26")
    ax1.plot(data27FB2.index, data27FB2.mean_essai_sem, 'o-', c = 'y', label = " moyenne des essais semaine 27")
    plt.xlabel("Heure d'essai")
    plt.ylabel("Moyenne_Fluo")
    ax1.set_xlim(-1, 180)
    #ax1.set_ylim(0,60)
    plt.title("Courbes moyenne Fluo des Semaines T3 2016 FB2")
    plt.legend()
    plt.show()
    plt.savefig(file_name +'/'+ 'Courbes moyenne Fluo des Semaines T3 2016 FB')    
semaine_plot()
#==============================================================================
"""---------------- graphique des semaines normalisées -------------------- """
#==============================================================================

def semaine_norm_plot():
    fig2, ax2 = plt.subplots(1, 1, figsize =(18,10))
    ax2.plot(data26FB2.index, data26FB2.moyenne_norm_par_essai, 'o-', c = 'blue', label = " moyenne des essais semaine 26")
    ax2.plot(data27FB2.index, data27FB2.moyenne_norm_par_essai, 'o-', c = 'y', label = " moyenne des essais semaine 27")   
    plt.xlabel("Heure d'essai ")
    plt.ylabel("Moyenne_Fluo")
    #ax1.set_xlim(-1, 180)
    #ax1.set_ylim(0, 2.5)
    plt.title("Courbes des Moyennes Normalisees des Semaines 1-22 2016 FB2")
    plt.legend()
    plt.show()
    plt.savefig(file_name +'/' + 'Courbes des Moyennes Normalisees des Semaines 1-22 2016 FB2') 
    return;
#==============================================================================
"""--------------- conca && discrétisation par interval------------------ """
#==============================================================================
dataRevian = dataR

def data_interval_1h_seuil():
    data = data.sort(["delta_to_hours"]) 
    data = data.reset_index( drop = True)    
    interval = []
    for i in xrange(len(data)):
        for j in range(int(data.delta_to_hours.max() + 1)):           
            if (  j  <= data.delta_to_hours[i] < j+1 ):
                interval.append(j+1)  
    pass
    data["interval"] = interval
    data = data[["mean_essai_sem","delta_to_hours","interval"]]
    return;
#==============================================================================
"""----------------------- Moyenne par intervalle """   
#==============================================================================
def moyenne_par_interval():
     m = (data[["mean_essai_sem", "interval"]].groupby(['interval']).mean()).values
     e = (data.interval.value_counts(sort = False).sort_index()).values
     moyenne_par_interval = []
     for i in xrange(len(e)): 
         tmp=np.repeat(m[i], e[i])
         moyenne_par_interval.extend(tmp)
         i = i + 1
     pass
     data["moyenne_par_interval"] = moyenne_par_interval
     return;
     
#==============================================================================
"""-------------- select une seule  fois chaque intervalle----------------- """     
#==============================================================================
def data_reduite_interval():
    Heure = [1]
    mean_par_interval = [data.moyenne_par_interval[0]]
    for i in xrange( len(data) - 1):
        if ((data.interval[i + 1 ] - data.interval[i]) >  0):
            Heure.append(data.interval[i])
            mean_par_interval.append(data.moyenne_par_interval[i])
        else:
            pass
    d = {'Heure': Heure , 'mean_par_interval': mean_par_interval}
    dataR13 = pd.DataFrame( data = d)
    return ;   
    
#==============================================================================
"""----------------- graphique moyenne && rolling par interval -------------"""
#==============================================================================
def data_interval_plot():
    #dataR.index = dataR.Heure
    fig3, ax3 = plt.subplots(1, 1, figsize =(18,10))
    ax3.plot(dataRevian.Heure, dataRevian.mean_par_interval, 'o-', c = 'blue', label ="Moyenne Total Evian ") 
    curve_10 = pd.rolling_mean(dataRevian["mean_par_interval"] , 10, min_periods = dataRevian.index[0])  
    curve_10.plot(label ='mean average 10 Evian', legend= True, c='r', linewidth=2.0) 
    #♪ewma = pd.ewma(dataR["mean_par_interval"] , span = 10)
    #ewma.plot(label ='ewma 10', legend= True, c='b', linewidth=2.0)
    ax3.plot(dataR13.Heure, dataR13.mean_par_interval, 'o-', c = 'orange', label ="Moyenne T3 par interval")
    ax3.axhline( 18.64 ,  c = 'darkorange', label = "moyenne d'entropie", alpha=1) 
    ax3.set_xlim(-0.2, 180 )
    ax3.set_ylim(5,30)
    plt.legend()
    plt.xlabel("Heure")
    plt.ylabel("Moyenne de la Fluo ")
    plt.title("Evian vs T3")
    plt.show()
    plt.savefig(file_name +'/'+"Evian vs T3")
    return; 
    
#==============================================================================
"""---------------- stocker en un fichier hdf5 ---------------------------  """    
#==============================================================================
def data_hdf_stock():
    stock=pd.HDFStore("data_FB2.h5")
    for data in [data01FB2, data10FB2, data21FB2FB2, data14FB2, data21, data22FB2]:
        stock.append(data)
  
#------------------------------------------------------------------------------
if __name__ == "__main__":
    moyenne_norm()
    semaine_plot()
    semaine_norm_plot()   
    data_interval_1h_seuil() 
    moyenne_par_interval()
    data_reduite_interval()
    data_interval_plot()






#prstd, iv_l, iv_u = wls_prediction_std(res)
#fig, ax = plt.subplots(figsize=(8,6))
#ax.plot(x, y, 'o', label="data")
#ax.plot(x, y_true, 'b-', label="True")
#ax.plot(x, res.fittedvalues, 'r--.', label="OLS")
#ax.plot(x, iv_u, 'r--')
#ax.plot(x, iv_l, 'r--')
#ax.legend(loc='best');

#path =r'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/obs_semaines/FB_evian_moyenne_TTsemaines/FB2_TT/' # use your path
#def data_concat():    
#    allFiles = glob.glob(path + "/*.csv")
#    data = pd.DataFrame()
#    list_ = []
#    for file_ in allFiles:
#        print file_
#        df = pd.read_csv(file_ , index_col = 0 , sep =";")
#        list_.append(df)
#    data = pd.concat(list_)
#    data[['hours_debut_essai']] = data[['hours_debut_essai']].astype(timedelta)
#    data = data.sort(["hours_debut_essai"])                         # ordoner par date
#    data = data.reset_index(drop = True) 
#    data.index = data["hours_debut_essai"].astype(timedelta)
#    return ;
#    
#format = '%d'+' day'+', %HH:%MM:%SS'
#print datetime.strptime(data.index[150], format) 
#    
#    
#delta = '1 days, 0:19:56.486000'
#days, hours, minutes, seconds = re.match('(?:(\d+) days or days, )?(\d+):(\d+):([.\d+]+)', delta).groups()
#total_seconds = ((int(days or 0) * 24 + int(hours)) * 60 + int(minutes)) * 60 + float(seconds)