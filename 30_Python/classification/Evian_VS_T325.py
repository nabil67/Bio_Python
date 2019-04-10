#-*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:51:37 2016
@author: nabil.belahrach
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics, neighbors, tree, ensemble, linear_model, cross_validation, svm
from sklearn.cross_validation import train_test_split 
import seaborn as sns
plt.style.use('ggplot')
import time as tm
import sklearn.neural_network as nne
from sklearn.naive_bayes import GaussianNB
import ffnet as ff



"""
# On récupère les semaine T3_FB1  { 13_2016, 41_2015, 43_2015, 46_2015 } ┼ on rajoutela dimension temporelle "date_relatif",
  et le tague classe = T3.
  
# On récupère la semaine Evian_FB1 { 01_2016, 10_2016, 11_2016, 14_2016, 21_2016, 22_2016 } + "date_relatif" + classe = Evian.
 Les données sont enregistrées dans /classification/comparaison_FB1_Evian-T3.spydata
"""



file_name = 'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/classification/'

def data_pretraitement():
    data_T3 = pd.concat([data13_2016FB1, data41_2015FB1, data43_2015FB1, data46_2015FB1], axis = 0, ignore_index= True)  
    data_T3["classe"] = "T3" 
    data_Evian = pd.concat([data01, data10, data11, data14, data21, data22], axis = 0, ignore_index = True)
    data_Evian["classe"]="Evian"
    data_T3 = data_T3[["classe","delta_to_hours","mean_essai_sem", "moyenne_norm_par_essai"]]
    data_Evian = data_Evian[["classe","delta_to_hours","mean_essai_sem", "moyenne_norm_par_essai"]]
    dfc = pd.concat([data_Evian, data_T3], axis = 0, ignore_index = True)
    #dfc = dfc[dfc["delta_to_hours"] > 48 ]
    return dfc;

#==============================================================================
"""-------------------- train_test_split-------------------------------------- """
#==============================================================================

def data_XY(dfc):
    Y = dfc["classe"].values
    X = dfc[["mean_essai_sem", "delta_to_hours", "moyenne_norm_par_essai"]].values
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    return X, Y;

#==============================================================================
"""----------------- cross_validation.StratifiedKFold ------------------------ """
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
    
"""-------------------- accuracY --------------------------------------------- """
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
    GBN_accuracy = metrics.accuracy_score(Y, stratified_cv(X, Y, GaussianNB))
    pass
    print('Passive Aggressive Classifier: {:.2f}'.format(pag_accuracy))
    print('Gradient Boosting Classifier:  {:.2f}'.format(gradBost_accuracy))
    print('Support vector machine(SVM):   {:.2f}'.format( svm_accuracy))
    print('Random Forest Classifier:      {:.2f}'.format(RDForest_accuracy))
    print('K Nearest Neighbor Classifier: {:.2f}'.format(knn_accuracy))
    print('Logistic Regression:           {:.2f}'.format(Reglogistic_accuracy))
    return pag_accuracy, tree_accuracy, GBN_accuracy,  svm_accuracy, gradBost_accuracy, ridge_accuracy,  RDForest_accuracy, knn_accuracy, Reglogistic_accuracy, adaboost_accuracy
    
"""--------------- precision, recall, f1-score -------------- """

def print_precision_recall_F1score():
    print('Passive Aggressive Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, linear_model.PassiveAggressiveClassifier))));
    print('Gradient Boosting Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, ensemble.GradientBoostingClassifier))));
    print('Support vector machine(SVM):\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, svm.SVC))));
    print('Random Forest Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, ensemble.RandomForestClassifier))));
    print('K Nearest Neighbor Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, neighbors.KNeighborsClassifier))));
    print('Logistic Regression:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, linear_model.LogisticRegression))));
    print('AdaBoost Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, ensemble.AdaBoostClassifier))));
    print('GaussianNB Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, GaussianNB))));

    #print('Dump Classifier:\n {}\n'.format(metrics.classification_report(y, [0 for ii in y.tolist()]))); # ignore the warning as they are all 0
    pass
    return;
#==============================================================================
"""---------------------matrix de confusion-------------------------------- """
#==============================================================================
def plot_confusion_matrix():
    k_neighbors_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, neighbors.KNeighborsClassifier))
    grad_ens_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, ensemble.GradientBoostingClassifier))
    svm_svc_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, svm.SVC))
#    pass_agg_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, linear_model.PassiveAggressiveClassifier))
#    random_forest_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, ensemble.RandomForestClassifier))
#    logistic_reg_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, linear_model.LogisticRegression))   
#    adaboost_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, ensemble.AdaBoostClassifier))
#    GaussianNB_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, GaussianNB))
#    decision_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, tree.DecisionTreeClassifier))
#    ridge_clf_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, linear_model.RidgeClassifier))
    pass
    conf_matrix = {    
                     1: {
                        'matrix': k_neighbors_conf_matrix,
                        'title': 'K Nearest Neighbors {:.2f} %'.format( knn_accuracy * 100),
                       },
                     2: {
                        'matrix': svm_svc_conf_matrix,
                        'title': 'Support Vector Machine {:.2f} %'.format(svm_accuracy * 100),
                       },
                       
                     3: {
                        'matrix': grad_ens_conf_matrix,
                        'title': 'Gradient Boosting {:.2f} %'.format(gradBost_accuracy * 100 ),
                       },                      
                                              
#                    4: {
#                        'matrix': GaussianNB_conf_matrix,
#                        'title': 'GaussianNB {:.2f} %'.format(pag_accuracy * 100),
#                       },
#
#                    5: {
#                        'matrix': decision_conf_matrix,
#                        'title': 'Decision Tree {:.2f} %'.format(tree_accuracy * 100),
#                       },
#                    6: {
#                        'matrix': ridge_clf_conf_matrix,
#                        'title': 'Ridge {:.2f} %'.format(ridge_accuracy * 100),
#                       },
#
#                    7: {
#                        'matrix': random_forest_conf_matrix,
#                        'title': 'Random Forest {:.2f} %'.format( RDForest_accuracy * 100),
#                       },
#
#                    8: {
#                        'matrix': logistic_reg_conf_matrix,
#                        'title': 'Logistic Regression {:.2f} %'.format(Reglogistic_accuracy * 100),
#                       },
#                    9: {
#                        'matrix': adaboost_conf_matrix,
#                        'title': 'AdaBoost {:.2f} %'.format(adaboost_accuracy * 100),
#                       },
    }
    
    fix, ax = plt.subplots(figsize=(12, 4))
    plt.suptitle('matrices de confusion des differents algorithmes')
    for ii, values in conf_matrix.items():
        matrix = values['matrix']
        title = values['title']
        plt.subplot(1, 3, ii) # starts from 1
        plt.title(title);
        sns.heatmap(matrix, annot=True,  fmt='');
    strtime = tm.strftime("%Y-%m-%d-%Hh%M")
    figure_name = "Confusion_Matrix_VC_2c {:}.png".format(strtime)
    plt.savefig(file_name + figure_name)
    plt.show()
    return;
#==============================================================================
"""--------------------- man() -------------------------------------------- """    
#==============================================================================

if __name__ == "__main__":
    dfc = data_pretraitement()
    X, Y = data_XY(dfc)
    pag_accuracy, GBN_accuracy, tree_accuracy, svm_accuracy, gradBost_accuracy, ridge_accuracy,  RDForest_accuracy, knn_accuracy, Reglogistic_accuracy, adaboost_accuracy = plot_accuracy()
    print_precision_recall_F1score()
    plot_confusion_matrix()
    
    
    

        