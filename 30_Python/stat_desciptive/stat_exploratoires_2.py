# -*- coding: utf-8 -*-
"""
Created on Mon May 09 11:17:23 2016

@author: nabil.belahrach
"""

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
    fig, axes = plt.subplots(3, 3, figsize =(10,16))
    values=pd.Series(df["moyenne"])
    df.boxplot(column="moyenne", by="classe",ax=axes[0][0])
#    values.hist(color='g',ax=axes[1], normed=True);
#    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True)
#    plt.savefig('kde_boxplot_moyenne.png')
#    plt.close()
    pass
    
    """ ecart-type """

    values=pd.Series(df["ecart-type"])
    df.boxplot(column="ecart-type", by="classe",ax=axes[0][1])
#    values.hist(color='g',ax=axes[1], normed=True);
#    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True)
#    plt.savefig('kde_boxplot_ecart-type.png')
#    plt.close()
    pass
    
    """ mediane """

    values=pd.Series(df["mediane"])
    df.boxplot(column="mediane", by="classe",ax=axes[0][2])
#    values.hist(color='g',ax=axes[1], normed=True)
#    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True)
#    plt.savefig('kde_boxplot_mediane.png')
#    plt.close()
    pass
    
    
    """ entropie """

    values=pd.Series(df["entropie"])
    df.boxplot(column="entropie", by="classe",ax=axes[1][0])
#    values.hist(color='g',ax=axes[1], normed=True)
#    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True)
#    plt.savefig('kde_boxplot_entropie.png')
#    plt.close()
    pass

    
     
    """ uniformit """

    values=pd.Series(df["uniformit"])
    df.boxplot(column="uniformit", by="classe",ax=axes[1][1])
#    values.hist(color='g',ax=axes[1], normed=True) ;
#    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True) 
#    plt.savefig('kde_boxplot_uniformit.png')
#    plt.close()
    pass
    
    
    """ surface """

    values=pd.Series(df["surface"])
    df.boxplot(column="surface", by="classe",ax=axes[1][2])
#    values.hist(color='g',ax=axes[1], normed=True) 
#    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True) 
#    plt.savefig('kde_boxplot_surface.png')
#    plt.close()
    pass
    """ eccent"""

    values=pd.Series(df["eccentricity"])
    df.boxplot(column="eccentricity", by="classe",ax=axes[2][0])
#    values.hist(color='g',ax=axes[1], normed=True)
#    values.plot(kind="KDE",ax=axes[1], style='r-', label=" KDE", legend= True)
#    plt.savefig('kde_boxplot_entropie.png')
#    plt.close()
    pass
    
    """ Andrews Curves & parallel coordinates ploting """

        
    fig, axes = plt.subplots(2,1, figsize =(10,17)) 
    andrews_curves(df2,'classe', ax=axes[0])
    parallel_coordinates(df2,'classe', ax=axes[1])
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
       
 boxplot_matrix = {
                1: {
                    'matrix': ax1,
                    'title': 'Passive Aggressive',
                   },
                2: {
                    'matrix': grad_ens_conf_matrix,
                    'title': 'Gradient Boosting',
                   },
                3: {
                    'matrix': decision_conf_matrix,
                    'title': 'Decision Tree',
                   },
                4: {
                    'matrix': ridge_clf_conf_matrix,
                    'title': 'Ridge',
                   },
                5: {
                    'matrix': svm_svc_conf_matrix,
                    'title': 'Support Vector Machine',
                   },
                6: {
                    'matrix': random_forest_conf_matrix,
                    'title': 'Random Forest',
                   },
                7: {
                    'matrix': k_neighbors_conf_matrix,
                    'title': 'K Nearest Neighbors',
                   },
                8: {
                    'matrix': logistic_reg_conf_matrix,
                    'title': 'Logistic Regression',
#                   }
#                9: {
#                    'matrix': dumb_conf_matrix,
#                    'title': 'Dumb',
                   },
}

fix, ax = plt.subplots(figsize=(16, 12))
plt.suptitle('Confusion Matrix of Various Classifiers')
for ii, values in conf_matrix.items():
    matrix = values['matrix']
    title = values['title']
    plt.subplot(3, 3, ii) # starts from 1
    plt.title(title);
    sns.heatmap(matrix, annot=True,  fmt='');
#plt.savefig('Confusion_Matrix_Various_Classifiers.png')
plt.show()      
       
#==============================================================================
# 
#==============================================================================
# coding: utf-8
