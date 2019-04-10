# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 14:34:32 2016

@author: Nabil.BELAHRACH
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, metrics, svm, tree, ensemble, linear_model, neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
import seaborn as sns
import time as tm
from IPython.display import Image
from sklearn.externals.six import StringIO


plt.style.use('ggplot')


#==============================================================================
# --------------------- données: Evian_VS_T3_2_C par images--------------------
#==============================================================================
"""
 T3_FB1:  { 13_2016, 41_2015, 43_2015, 46_2015 },
 et le tague classe = T3.
  
 Evian_FB1: { 01_2016, 10_2016, 11_2016, 14_2016, 21_2016, 22_2016 } 
 
 On classifier par images  T3[13_2016] vs Evian[10_2016, 11_2016, 22_2016] 
"""
#==============================================================================
# ----------------------------- Agregation -----------------------------------
#==============================================================================

def df_agregation():
    df13["classe"] = "T3" 
    for df in [df10, df11, df22]:
        df["classe"] = "Evian"
    pass
    data = pd.concat([df10, df11, df22, df13], ignore_index = True )
    data = data[["classe","moyenne", "ecart-type", "solidite","entropie", "smoothness","mm_gradient","num_essai"]]
    return data;

    
#==============================================================================
# --------------------------- X Y ---------------------------------------------   
#==============================================================================
def data_XY():
    Y = data["classe"].values
    X = data[["moyenne", "ecart-type", "solidite","entropie", "smoothness","mm_gradient","num_essai"]].values
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    return  X, Y
  
#==============================================================================
"""----------------- cross_validation.StratifiedKFold ------------------------ """
#==============================================================================
def stratified_cv(X, Y, clf_class, shuffle=True, n_folds=5, **kwargs):
    stratified_k_fold = cross_validation.StratifiedKFold(Y, n_folds=n_folds, shuffle=shuffle)
    Y_pred = Y.copy()
    for ii, jj in stratified_k_fold:
        X_train, X_test = X[ii], X[jj]
        Y_train = Y[ii]
        clf = clf_class(**kwargs)
        clf.fit(X_train,Y_train)
        Y_pred[jj] = clf.predict(X_test)
    return Y_pred

#==============================================================================
"""-------------------- accuracY ------------------------------------------ """   
#==============================================================================

def plot_accuracy():  
    svm_accuracy = metrics.accuracy_score(Y, stratified_cv(X, Y, svm.SVC))
    gradBost_accuracy = metrics.accuracy_score(Y, stratified_cv(X, Y, ensemble.GradientBoostingClassifier))
    knn_accuracy = metrics.accuracy_score(Y, stratified_cv(X, Y, neighbors.KNeighborsClassifier))
    Reglogistic_accuracy = metrics.accuracy_score(Y, stratified_cv(X, Y, linear_model.LogisticRegression))
    pass
    print('Gradient Boosting Classifier:  {:.2f}'.format(gradBost_accuracy))
    print('Support vector machine(SVM):   {:.2f}'.format( svm_accuracy))
    print('K Nearest Neighbor Classifier: {:.2f}'.format(knn_accuracy))
    print('Logistic Regression:           {:.2f}'.format(Reglogistic_accuracy))
    return   svm_accuracy, gradBost_accuracy,  knn_accuracy, Reglogistic_accuracy;
    
"""--------------------- precision, recall, f1-score -------------------- """

#def print_precision_recall_F1score():
#    print('Passive Aggressive Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, linear_model.PassiveAggressiveClassifier))));
#    print('Gradient Boosting Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, ensemble.GradientBoostingClassifier))));
#    print('Support vector machine(SVM):\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, svm.SVC))));
#    print('Random Forest Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, ensemble.RandomForestClassifier))));
#    print('K Nearest Neighbor Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, neighbors.KNeighborsClassifier))));
#    print('Logistic Regression:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, linear_model.LogisticRegression))));
#    print('AdaBoost Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, ensemble.AdaBoostClassifier))));
#    print('GaussianNB Classifier:\n {}\n'.format(metrics.classification_report(Y, stratified_cv(X, Y, GaussianNB))));
#
#    #print('Dump Classifier:\n {}\n'.format(metrics.classification_report(y, [0 for ii in y.tolist()]))); # ignore the warning as they are all 0
#    pass
#    return;
    
#==============================================================================
"""---------------------matrix de confusion-------------------------------- """
#==============================================================================
def plot_confusion_matrix():
    grad_ens_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, ensemble.GradientBoostingClassifier))
    svm_svc_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, svm.SVC))
    k_neighbors_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, neighbors.KNeighborsClassifier))
    logistic_reg_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, linear_model.LogisticRegression))   
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
                     4: {
                        'matrix': logistic_reg_conf_matrix,
                        'title': 'Logistic Regression {:.2f} %'.format(Reglogistic_accuracy * 100),
                       }                 
    }
    
    fix, ax = plt.subplots(figsize=(12, 8))
    plt.suptitle('matrices de confusion des differents algorithmes')
    for ii, values in conf_matrix.items():
        matrix = values['matrix']
        title = values['title']
        plt.subplot(2, 3, ii) # starts from 1
        plt.title(title);
        sns.heatmap(matrix, annot=True,  fmt='');
    strtime = tm.strftime("%Y%m%d_%Hh%M")
    figure_name = "Confusion_Matrix_VC_2c {:}.png".format(strtime)
    plt.savefig(file_name + figure_name)
    plt.show()
    return;   
    
#==============================================================================
"""----------------------- features importances ------------------------- """
#==============================================================================

def GB_features_importances():
    gbc = ensemble.GradientBoostingClassifier()
    gbc.fit(X, Y)
    feature_importance = gbc.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(16, 12))
    plt.barh(pos, feature_importance[sorted_idx], align='center', color='#7A68A6')
    plt.yticks(pos, np.asanyarray(data.columns[1:].tolist())[sorted_idx])
    plt.xlabel('GBoosting Relative Importance')
    plt.title('Variable Importance')
    strtime = tm.strftime("%Y%m%d_%Hh%M")
    plt.savefig(file_name + 'Relative_Importance_GBoosting_2c_{:}'.format(strtime))
    pass
    plt.show()
    return;
#==============================================================================
"""----------------------- tree exporting ------------------------- """
#==============================================================================
#def exporting_tree():
#    clf = tree.DecisionTreeClassifier()
#    clf = clf.fit(X, Y)
#    dot_data = StringIO()
#    tree.export_graphviz(clf, out_file=dot_data,  
#                         filled=True, rounded=True,  
#                         special_characters=True) 
#                         
#    graph = pydot.graph_from_dot_data(dot_data.getvalue())
#    Image(graph.create_png())  
#    return;
#==============================================================================
"""------------------- main ----------------------------------------------- """    
#==============================================================================
if __name__ == "__main__":
    t0 = tm.time()
    file_name = 'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/classification/'
    data = df_agregation()
    X, Y = data_XY()
    svm_accuracy, gradBost_accuracy, knn_accuracy, Reglogistic_accuracy = plot_accuracy()
    plot_confusion_matrix()    
    GB_features_importances()
    t1 = tm.time()
    print("totale = ", (t1 - t0)/60)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
