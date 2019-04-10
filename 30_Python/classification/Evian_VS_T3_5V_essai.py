#-*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:51:37 2016
@author: nabil.belahrach
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, metrics, neighbors, tree, ensemble, linear_model, cross_validation, svm
from sklearn.cross_validation import train_test_split 
import seaborn as sns
plt.style.use('ggplot')
import time as tm
import sklearn.neural_network as nne
from sklearn.naive_bayes import GaussianNB




"""
# On récupère les semaine T3_FB1  { 13_2016, 41_2015, 43_2015, 46_2015 } ┼ on rajoutela dimension temporelle "date_relative",
  et les taguer classe = T3.
  
# On récupère la semaine Evian_FB1 { 01_2016, 10_2016, 11_2016, 14_2016, 21_2016, 22_2016 } + "date_relative" + classe = Evian.
 Les données sont enregistrées dans /classification/data_Evian_VS_T3_FB1_5V.spydata
"""

file_name = 'U:/Stagiaires/Nabil.BELAHRACH/Donnees/40_figures/classification/'

#==============================================================================
"""----------------------variables normalisées relatives---------------------- """
#==============================================================================

def moyenne_norm():    """ utilisée une seule fois pour rajouter les variables normalisées """
    for data in [data31FB1_2016_T3, data13FB1_2016_T3, data41FB1_2015_T3, data43FB1_2015_T3, data46FB1_2015_T3,data01Evian, data10EVian, data11Evian, data14Evian, data21EVian, data22Evian]:       
        moy_norm_essai = data.mean_essai_sem / data.mean_essai_sem[0]
        entropie_norm_essai = data.entropie_essai_sem /data.entropie_essai_sem[0]
        solidite_norm_essai = data.solidite_essai_sem / data.solidite_essai_sem[0]
        data["moy_norm_essai"] = moy_norm_essai
        data["entropie_norm_essai"]= entropie_norm_essai
        data["solidite_norm_essai"] = solidite_norm_essai     
        #data["semaine"] =  "data"
        #del data["semaine"]
    return; 

#==============================================================================
"""-----------------------------newData apres SMOTE-ing des donnees-------------"""
#==============================================================================
def data_smote():
    #data = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/classification/newData.csv", sep = ",")
    #data = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/30_Python/classification/data_classification_3c_6V_20160913_16h08.csv", sep = ",")
    data = data[["classe","delta_to_hours","mean_essai_sem","moy_norm_essai", "entropie_essai_sem","entropie_norm_essai", "sigma_essai_sem","solidite_essai_sem", "solidite_norm_essai"]]
    #☻data = data[data["delta_to_hours"] > 24 ]
    return data;
#==============================================================================
"""--------------------- Pétraitement ------------------------------------- """
#==============================================================================

def data_pretraitement():
    data_T3 = pd.concat([data13FB1_2016_T3, data41FB1_2015_T3, data43FB1_2015_T3, data46FB1_2015_T3], axis = 0, ignore_index= True)  
    data_T3["classe"] = "T3_325" 
    pass
    data_Evian = pd.concat([data01Evian, data10EVian, data11Evian, data14Evian, data21EVian, data22Evian], axis = 0, ignore_index = True)
    data_Evian["classe"]="Evian"
    data_T3_325 = data_T3[["semaine_et_FB","classe","delta_to_hours","mean_essai_sem", "moy_norm_essai","entropie_essai_sem","entropie_norm_essai","sigma_essai_sem","solidite_essai_sem","solidite_norm_essai"]]
    data_Evian = data_Evian[["semaine_et_FB","classe","delta_to_hours","mean_essai_sem", "moy_norm_essai", "entropie_essai_sem","entropie_norm_essai","sigma_essai_sem","solidite_essai_sem", "solidite_norm_essai"]]
    data31FB1_2016_T3["classe"] = "T3_225" 
    data_T3_225 = data31FB1_2016_T3[["semaine_et_FB","classe","delta_to_hours","mean_essai_sem", "moy_norm_essai", "entropie_essai_sem","entropie_norm_essai","sigma_essai_sem","solidite_essai_sem", "solidite_norm_essai"]]    
    data = pd.concat([data_Evian, data_T3_325, data_T3_225], axis = 0, ignore_index = True)
    data = data[["classe","semaine_et_FB","delta_to_hours","mean_essai_sem","moy_norm_essai", "entropie_essai_sem","entropie_norm_essai", "sigma_essai_sem","solidite_essai_sem", "solidite_norm_essai"]]
    #data = data[data["delta_to_hours"] > 24 ]
    return data;

#==============================================================================
"""-------------------- train_test_split-------------------------------------- """
#==============================================================================

def data_XY(data):
    Y = data["classe"].values
    X = data[["delta_to_hours","mean_essai_sem","moy_norm_essai", "entropie_essai_sem","entropie_norm_essai", "sigma_essai_sem","solidite_essai_sem"]].values
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    return X, Y, scaler;

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
    #pass_agg_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, linear_model.PassiveAggressiveClassifier))
    grad_ens_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, ensemble.GradientBoostingClassifier))
    decision_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, tree.DecisionTreeClassifier))
    ridge_clf_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, linear_model.RidgeClassifier))
    svm_svc_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, svm.SVC))
    random_forest_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, ensemble.RandomForestClassifier))
    k_neighbors_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, neighbors.KNeighborsClassifier))
    logistic_reg_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, linear_model.LogisticRegression))   
    adaboost_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, ensemble.AdaBoostClassifier))
    GaussianNB_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, GaussianNB))
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
                        'matrix': random_forest_conf_matrix,
                        'title': 'Random Forest {:.2f} %'.format( RDForest_accuracy * 100),

                       },

                    5: {
                        'matrix': adaboost_conf_matrix,
                        'title': 'AdaBoost {:.2f} %'.format(adaboost_accuracy * 100),
                       },
                    6: {
                        'matrix': ridge_clf_conf_matrix,
                        'title': 'Ridge {:.2f} %'.format(ridge_accuracy * 100),
                       },

                    7: {
                        'matrix': GaussianNB_conf_matrix,
                        'title': 'GaussianNB {:.2f} %'.format(pag_accuracy * 100),

                       },

                    8: {
                        'matrix': logistic_reg_conf_matrix,
                        'title': 'Logistic Regression {:.2f} %'.format(Reglogistic_accuracy * 100),
                       },
                    9: {
                        'matrix': decision_conf_matrix,
                        'title': 'Decision Tree {:.2f} %'.format(tree_accuracy * 100),
                       },
    }
    
    fix, ax = plt.subplots(figsize=(12, 12))
    plt.suptitle('matrices de confusion apres 0h ')
    for ii, values in conf_matrix.items():
        matrix = values['matrix']
        title = values['title']
        plt.subplot(3, 3, ii) # starts from 1
        plt.title(title);
        sns.heatmap(matrix, annot=True,  fmt='');
    strtime = tm.strftime("%Y-%m-%d-%Hh%M")
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
    plt.xlabel('GBoosting Relative Importance sans smote ')
    plt.title('Variable Importance')
    strtime = tm.strftime("%Y%m%d_%Hh%M")
    plt.savefig(file_name + 'Relative_Importance_GBoosting_3c_{:}'.format(strtime))
    pass
    plt.show()
    return;
    
#==============================================================================
"""----------------------- Y_false localisation ------------------------- """ 
#==============================================================================
def y_false_localisation():
    Y_pred = stratified_cv(X, Y, neighbors.KNeighborsClassifier)
    #Y_pred = stratified_cv(X, Y, ensemble.RandomForestClassifier)
    #Y_pred = stratified_cv(X, Y, ensemble.GradientBoostingClassifier)
    X_false = X[ Y != Y_pred ]
    X_true = X[ Y == Y_pred ]
    X_false_t = scaler.inverse_transform(X_false)
    X_true_t  = scaler.inverse_transform(X_true)
    fig3, ax3 = plt.subplots(1, 1, figsize =(18,10))
    ax3.plot(X_false_t[:,0], X_false_t[:,1], '>', c='red' ,markersize=10, label ="essais mal-classes") 
    ax3.plot(X_true_t[:,0], X_true_t[:,1], 'o', c = 'green', label = " essais bien-classes") 
    plt.xlabel("Heure")
    plt.ylabel("Moyenne Fluo")
    ax3.set_xlim(-1, )
    plt.legend()
    strtime = tm.strftime("%Y%m%d_%Hh%M")
    plt.title("KNN apres 24h Y_false = {:}".format(len(X_false)))
    plt.savefig(file_name + 'KNN apres 24h Y_false_localiation_3c_{:}'.format(strtime))
    plt.show()
    return;

    
#==============================================================================
"""--------------------- main-------------------------------------------- """    
#==============================================================================

if __name__ == "__main__":
    t0 = tm.time()
    #moyenne_norm()
    data = data_pretraitement()
    #data = data_smote()
    X, Y, scaler = data_XY(data)
    pag_accuracy, GBN_accuracy, tree_accuracy, svm_accuracy, gradBost_accuracy, ridge_accuracy,  RDForest_accuracy, knn_accuracy, Reglogistic_accuracy, adaboost_accuracy = plot_accuracy()
    #print_precision_recall_F1score()    pas besoin
    plot_confusion_matrix()
    GB_features_importances() 
    y_false_localisation()
    t1 = tm.time()
    print("time ="), (t1 - t0)
    
    
    
#data.to_csv("data_classification_3c_6V_{:}.csv".format(strtime), sep =";")
#def score(X,y): 
#
#       y1=lm.stratified_cvknn(X,y)
#
#       print y1
#
#       Xv=X
#
#       Xv["y"]=y
#
#       Xv["ypred"]=y1
#
#       Xv=Xv[Xv.y != Xv.ypred]
#
#       Xv["NumeroM"]=Xv.index
#
#       Xv=pd.merge(Xv,df1,on="NumeroM",how="right")
#
#       Xv=Xv[["NumeroM","Echantillon","y","ypred"]]
#
#       Xv=Xv.dropna()
#
#       counts = Xv.groupby(['NumeroM','Echantillon','y','ypred']).size()
#
#       counts=counts.to_frame()
#
#       print counts

#       print('Ridge:{:.2f}'.format(metrics.accuracy_score(y, y1))) 