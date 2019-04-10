# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 15:26:46 2016

@author: nabil.belahrach
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import neighbors
from sklearn import tree
import seaborn as sns
import time as tm
plt.style.use('ggplot')
#plt.style.use('fivethirtyeight') # Good looking plots
#pd.set_option('display.max_columns', None)
#from Mesfonctions import *


df = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/20_brut/VT_20141030_EvryOriginales_entropie.csv", 
                   sep=";", header = False,usecols=[1,14,15,16,19,20,21,26],
                   names = ['classe','moyenne','ecart-type','mediane','entropie','uniformit','surface','eccentricity' ])
                   
                   
file_name = 'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/classification/'

#==============================================================================
"""---------------------------Préparation des données ------------------ """
#==============================================================================
def df_preparation():
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
    return df
#df2 = df[['classe', 'uniformit','surface','eccentricity']]



def data_XY(dfc):
    Y = dfc["classe"].values
    X = dfc[["moyenne","ecart-type","mediane","entropie","uniformit","surface","eccentricity"]].values
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    return X, Y;

#==============================================================================
"""------------ cross_validation.StratifiedKFold -------------- """
#==============================================================================

def stratified_cv(X, Y, clf_class, shuffle=True, n_folds=10, **kwargs):
    stratified_k_fold = cross_validation.StratifiedKFold(Y, n_folds=n_folds, shuffle=shuffle)
    Y_pred = Y.copy()
    for ii, jj in stratified_k_fold:
        X_train, X_test = X[ii], X[jj]
        Y_train = Y[ii]
        clf = clf_class(**kwargs)
        clf.fit(X_train,Y_train)
        Y_pred[jj] = clf.predict(X_test)
    return Y_pred
    
"""------------------ accuracY --------------------------------------------- """
def plot_accuracy():  # a modifier    
    pag_accuracy = metrics.accuracy_score(Y,stratified_cv(X, Y, linear_model.PassiveAggressiveClassifier))
    svm_accuracy = metrics.accuracy_score(Y, stratified_cv(X, Y, svm.SVC))
    gradBost_accuracy = metrics.accuracy_score(Y, stratified_cv(X, Y, ensemble.GradientBoostingClassifier))
    ridge_accuracy = metrics.accuracy_score(Y, stratified_cv(X, Y, linear_model.RidgeClassifier))
    RDForest_accuracy = metrics.accuracy_score(Y, stratified_cv(X, Y, ensemble.RandomForestClassifier))
    knn_accuracy = metrics.accuracy_score(Y, stratified_cv(X, Y, neighbors.KNeighborsClassifier))
    Reglogistic_accuracy = metrics.accuracy_score(Y, stratified_cv(X, Y, linear_model.LogisticRegression))
    tree_accuracy = metrics.accuracy_score(Y, stratified_cv(X, Y, tree.DecisionTreeClassifier))
    adaboost_accuracy = metrics.accuracy_score(Y, stratified_cv(X, Y, ensemble.AdaBoostClassifier))
    pass
    print('Passive Aggressive Classifier: {:.2f}'.format(pag_accuracy))
    print('Gradient Boosting Classifier:  {:.2f}'.format(gradBost_accuracy))
    print('Support vector machine(SVM):   {:.2f}'.format( svm_accuracy))
    print('Random Forest Classifier:      {:.2f}'.format(RDForest_accuracy))
    print('K Nearest Neighbor Classifier: {:.2f}'.format(knn_accuracy))
    print('Logistic Regression:           {:.2f}'.format(Reglogistic_accuracy))
    return pag_accuracy, tree_accuracy,  svm_accuracy, gradBost_accuracy, ridge_accuracy,  RDForest_accuracy, knn_accuracy, Reglogistic_accuracy, adaboost_accuracy
    
"""--------------- precision, recall, f1-score -------------- """

def print_precision_recall_F1score():
    print('Passive Aggressive Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, linear_model.PassiveAggressiveClassifier))));
    print('Gradient Boosting Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, ensemble.GradientBoostingClassifier))));
    print('Support vector machine(SVM):\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, svm.SVC))));
    print('Random Forest Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, ensemble.RandomForestClassifier))));
    print('K Nearest Neighbor Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, neighbors.KNeighborsClassifier))));
    print('Logistic Regression:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, linear_model.LogisticRegression))));
    print('AdaBoost Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, ensemble.AdaBoostClassifier))));
    #print('Dump Classifier:\n {}\n'.format(metrics.classification_report(y, [0 for ii in y.tolist()]))); # ignore the warning as they are all 0
    pass
    return;
    
#==============================================================================
"""---------------------matrix de confusion-------------------------------- """
#==============================================================================
def plot_confusion_matrix():
    pass_agg_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, linear_model.PassiveAggressiveClassifier))
    grad_ens_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, ensemble.GradientBoostingClassifier))
    decision_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, tree.DecisionTreeClassifier))
    ridge_clf_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, linear_model.RidgeClassifier))
    svm_svc_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, svm.SVC))
    random_forest_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, ensemble.RandomForestClassifier))
    k_neighbors_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, neighbors.KNeighborsClassifier))
    logistic_reg_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, linear_model.LogisticRegression))   
    adaboost_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, ensemble.AdaBoostClassifier))
    pass
    conf_matrix = {
                    1: {
                        'matrix': pass_agg_conf_matrix,
                        'title': 'Passive Aggressive {:.2f} %'.format(pag_accuracy * 100),
                       },
                    2: {
                        'matrix': grad_ens_conf_matrix,
                        'title': 'Gradient Boosting {:.2f} %'.format(gradBost_accuracy * 100 ),
                       },
                    3: {
                        'matrix': decision_conf_matrix,
                        'title': 'Decision Tree {:.2f} %'.format(tree_accuracy * 100),
                       },
                    4: {
                        'matrix': ridge_clf_conf_matrix,
                        'title': 'Ridge {:.2f} %'.format(ridge_accuracy * 100),
                       },
                    5: {
                        'matrix': svm_svc_conf_matrix,
                        'title': 'Support Vector Machine {:.2f} %'.format(svm_accuracy * 100),
                       },
                    6: {
                        'matrix': random_forest_conf_matrix,
                        'title': 'Random Forest {:.2f} %'.format( RDForest_accuracy * 100),
                       },
                    7: {
                        'matrix': k_neighbors_conf_matrix,
                        'title': 'K Nearest Neighbors {:.2f} %'.format( knn_accuracy * 100),
                       },
                    8: {
                        'matrix': logistic_reg_conf_matrix,
                        'title': 'Logistic Regression {:.2f} %'.format(Reglogistic_accuracy * 100),
                       },
                    9: {
                        'matrix': adaboost_conf_matrix,
                        'title': 'AdaBoost {:.2f} %'.format(adaboost_accuracy * 100),
                       },
    }
    
    fix, ax = plt.subplots(figsize=(16, 12))
    plt.suptitle('matrices de confusion des differents algorithmes')
    for ii, values in conf_matrix.items():
        matrix = values['matrix']
        title = values['title']
        plt.subplot(3, 3, ii) # starts from 1
        plt.title(title);
        sns.heatmap(matrix, annot=True,  fmt='');
    strtime = tm.strftime("%Y-%m-%d-%H-%M")
    figure_name = "Confusion_Matrix_VC_2c {:}.png".format(strtime)
    plt.savefig(file_name + figure_name)
    plt.show()
    return;
    
#==============================================================================
"""--------------------- man() -------------------------------------------- """    
#==============================================================================

if __name__ == "__main__":
    dfc = df_preparation()
    X, Y = data_XY(dfc)
    pag_accuracy, tree_accuracy, svm_accuracy, gradBost_accuracy, ridge_accuracy,  RDForest_accuracy, knn_accuracy, Reglogistic_accuracy, adaboost_accuracy = plot_accuracy()
    print_precision_recall_F1score()
    plot_confusion_matrix()
    