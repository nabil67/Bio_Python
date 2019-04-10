# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 09:32:05 2016

@author: nabil.belahrach
"""

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')                                    # pour ne pas affcher les graphes
import matplotlib.pylab as plt
from pandas.tools.plotting import andrews_curves
from pandas.tools.plotting import parallel_coordinates
from sklearn.decomposition import PCA
from pandas.tools.plotting import radviz
from pandas.tools.plotting import scatter_matrix
import statsmodels.api as sm

plt.style.use('ggplot')

filename ="U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures"



df = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/20_brut/VT_20141030_EvryOriginales_entropie.csv", 
                   sep=";", header = False,usecols=[1,14,15,16,19,20,21,26],
                   names = ['classe','moyenne','ecart-type','mediane','entropie','uniformit','surface','eccentricity' ])
                   
                   
#print df.head()                   
#df.shape

#==============================================================================
"""---------------------------Préparation des données -------------------- """
#==============================================================================

df['moyenne']=df['moyenne'].str.replace("," , ".")
df['entropie']=df['entropie'].str.replace("," , ".")
df['eccentricity']=df['eccentricity'].str.replace("," , ".")
df['uniformit']=df['uniformit'].str.replace("," , ".")
df['ecart-type']=df['ecart-type'].str.replace("," , ".")
df["classe"] =pd.Categorical(df["classe"])

cl = pd.CategoricalIndex(df["classe"]).categories
df[['moyenne','ecart-type','mediane','surface','entropie','uniformit','eccentricity']] = df[['moyenne','ecart-type',
                                          'mediane','surface','entropie','uniformit','eccentricity']].astype(float)
df["classe"] = df["classe"].cat.rename_categories(["c2","c3","c6","c4","c5","c1"])

df["classe"] = df.classe.cat.reorder_categories(["c1","c2","c3","c4","c5","c6"])

#g1=df[df.classe == "c1"] 
#g2=df[df.classe == "c2"] 
#g3=df[df.classe == "c3"] 
#g4=df[df.classe == "c4"] 
#g5=df[df.classe == "c5"] 
#g6=df[df.classe == "c6"] 
#
#dfq =pd.concat([g1,g2,g3,g4,g5,g6],axis=0,ignore_index=True)
#df.to_csv('U:/Stagiaires/Nabil.BELAHRACH/Donnees/20_brut/myFile_entropie.csv', sep = ';')

#==============================================================================
"""---------------------KDE_boxplot des variables/classe -------------------"""
#==============================================================================
def plot_main():    
    """ moyenne """
    fig, axes = plt.subplots(2, 4, figsize =(10,16))
    values=pd.Series(df["moyenne"])
    df.boxplot(column="moyenne", by="classe",ax=axes[0][0])
    values.hist(color='g',ax=axes[1], normed=True);
    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True)
#    plt.savefig('kde_boxplot_moyenne.png')
#    plt.close()
    pass
    
    """ ecart-type """
    fig, axes = plt.subplots(2, 1, figsize =(10,16))
    values=pd.Series(df["ecart-type"])
    df.boxplot(column="ecart-type", by="classe",ax=axes[0][1])
    values.hist(color='g',ax=axes[1], normed=True);
    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True)
    plt.savefig('kde_boxplot_ecart-type.png')
    plt.close()
    pass
    
    """ mediane """
    fig, axes = plt.subplots(2, 1, figsize =(10,16))
    values=pd.Series(df["mediane"])
    df.boxplot(column="mediane", by="classe",ax=axes[0][2])
    values.hist(color='g',ax=axes[1], normed=True)
    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True)
    plt.savefig('kde_boxplot_mediane.png')
    plt.close()
    pass
    
    
    """ entropie """
    fig, axes = plt.subplots(2, 1, figsize =(10,16))
    values=pd.Series(df["entropie"])
    df.boxplot(column="entropie", by="classe",ax=axes[0][3])
    values.hist(color='g',ax=axes[1], normed=True)
    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True)
    plt.savefig('kde_boxplot_entropie.png')
    plt.close()
    pass

    
     
    """ uniformit """
    fig, axes = plt.subplots(2, 1, figsize =(10,16))
    values=pd.Series(df["uniformit"])
    df.boxplot(column="uniformit", by="classe",ax=axes[1][0])
    values.hist(color='g',ax=axes[1], normed=True) ;
    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True) 
    plt.savefig('kde_boxplot_uniformit.png')
    plt.close()
    pass
    
    
    """ surface """
    fig, axes = plt.subplots(2, 1, figsize =(10,16))
    values=pd.Series(df["surface"])
    df.boxplot(column="surface", by="classe",ax=axes[1][1])
    values.hist(color='g',ax=axes[1], normed=True) 
    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True) 
    plt.savefig('kde_boxplot_surface.png')
    plt.close()
    pass
    """ eccent"""
    fig, axes = plt.subplots(2, 1, figsize =(10,16))
    values=pd.Series(df["eccentricity"])
    df.boxplot(column="eccentricity", by="classe",ax=axes[1][2])
    values.hist(color='g',ax=axes[1], normed=True)
    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True)
    plt.savefig('kde_boxplot_eccentricitie.png')
    plt.close()
    pass
    
    """ Andrews Curves & parallel coordinates ploting """

        
    fig, axes = plt.subplots(2,1, figsize =(10,17)) 
    andrews_curves(df,'classe', ax=axes[0])
    parallel_coordinates(df,'classe', ax=axes[1])
    plt.savefig('Andrew_curves_df2.png')

    plt.close()
    
    """ matrice des correlations  """
    fig, axes = plt.subplots(1,1, figsize =(10,10))
    scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde', ax = axes)
    plt.savefig('matrix_corr_kde.png')
    plt.close()
    pass
    
    """ RadViz visualizing multi-variate data"""
    fig, axes = plt.subplots(1,1, figsize =(10,10))
    radviz(df, "classe", ax = axes)
    plt.savefig('RadViz_df.png')  
    plt.close()
#    pass

    
if __name__ == '__main__':
    plot_main()





""" plot iteratif"""
#def data_plot(data):
#for i in range(5) :     
#        print df.ix[:,i].name
#       fig, axes = plt.subplots(2, 1, figsize =(10,16))       
#       values=pd.Series(df.columns)
#       print values
#       values.boxplot(column="uniformit", by="classe",ax=axes[0])
#       values.hist(color='g',ax=axes[1], normed=True)
#       values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True)
#       plt.savefig("str(element).png")
       

     
     