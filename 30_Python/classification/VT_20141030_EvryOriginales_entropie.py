# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:39:01 2016

@author: nabil.belahrach

# -*- coding: utf-8 -*-

Éditeur de Spyder

Ceci est un script temporaire.
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
#from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.metrics import classification_report
#from sklearn.decomposition import PCA
#import statsmodels.api as sm
from sklearn import metrics
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import neighbors
from sklearn import tree
import seaborn as sns
plt.style.use('ggplot')
#plt.style.use('fivethirtyeight') # Good looking plots
#pd.set_option('display.max_columns', None)
from Mesfonctions import *


df = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/20_brut/VT_20141030_EvryOriginales_entropie.csv", 
                   sep=";", header = False,usecols=[1,14,15,16,19,20,21,26],
                   names = ['classe','moyenne','ecart-type','mediane','entropie','uniformit','surface','eccentricity' ])
                   
                   
print df.head()                   
df.shape

#==============================================================================
"""---------------------------Préparation des données ------------------ """
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
#df2 = df[['classe', 'uniformit','surface','eccentricity']]

#df.to_csv('U:/Stagiaires/Nabil.BELAHRACH/Donnees/20_brut/myFile_entropie.csv', sep = ';')

#==============================================================================
"""---------------------- X, y -------------------------------------------- """
#==============================================================================


X = (df[:][['moyenne','ecart-type','mediane','surface','entropie','uniformit','eccentricity']]).values
#X = (df[:][['surface','uniformit','eccentricity']]).values
Y = (df[:][['classe']]).values
y= Y.ravel()
#df.classe.value_counts()

scaler= preprocessing.StandardScaler().fit(X)  
X = scaler.transform(X)

#==============================================================================
""" ------------------------ Classification KNN --------------------------- """
#==============================================================================

""" ------center et réduire les variables explicatives ! -----------"""

#polynomial_features = preprocessing.PolynomialFeatures()
#X = polynomial_features.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split( X, y, 
                                                 test_size= 0.25,random_state=33)

               # entrainement 
param = [{"n_neighbors": list(range(1,15))}]
knn = GridSearchCV(KNeighborsClassifier(), param, cv=10, n_jobs= -1) 
digit_knn = knn.fit(X_train, Y_train)  
print ("le best param = "), digit_knn.best_params_["n_neighbors"]       
pass                       # best param = 9

""" ------on relance le modèle avec le best-paramètre -----------"""

knn = KNeighborsClassifier(n_neighbors= digit_knn.best_params_["n_neighbors"])

digit_knn.score(X_train,Y_train)     # estimation de l'erreur = 55%
Y_pred = digit_knn.predict(X_test)   # prediction des réponses de X_test
table = pd.crosstab( Y_test, Y_pred) # matrice de confusion
print table;
print classification_report( Y_test, Y_pred)


def Acccuracy3diag(table):
    mat = table.values
    bc = 0
    for i in [-1,0,1]:        
        diag= np.trace(mat,offset= i )
        bc += diag 
    total = sum(sum(mat[:,:]))
    prc = float(bc)/total
    return prc
    
Acccuracy3diag(table)


plt.matshow(table)
plt.title("Matrice de Confusion du knn 3c")   # pas top
plt.colorbar()
plt.show()

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
"""------------------ accuracy --------------------------------------------- """

print('Passive Aggressive Classifier: {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, linear_model.PassiveAggressiveClassifier))))
print('Gradient Boosting Classifier:  {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))))
print('Support vector machine(SVM):   {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, svm.SVC))))
print('Random Forest Classifier:      {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, ensemble.RandomForestClassifier))))
print('K Nearest Neighbor Classifier: {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))))
print('Logistic Regression:           {:.2f}'.format(metrics.accuracy_score(y, stratified_cv(X, y, linear_model.LogisticRegression))))

"""--------------- precision, recall, f1-score -------------- """

print('Passive Aggressive Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, linear_model.PassiveAggressiveClassifier))));
print('Gradient Boosting Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))));
print('Support vector machine(SVM):\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, svm.SVC))));
print('Random Forest Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, ensemble.RandomForestClassifier))));
print('K Nearest Neighbor Classifier:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))));
print('Logistic Regression:\n {}\n'.format(metrics.classification_report(y, stratified_cv(X, y, linear_model.LogisticRegression))));
#print('Dump Classifier:\n {}\n'.format(metrics.classification_report(y, [0 for ii in y.tolist()]))); # ignore the warning as they are all 0
pass
#==============================================================================
"""----------------------- features importances ------------------------- """
#==============================================================================

"""--------------- selection de variables Grad_Bossiting -------------- """

gbc = ensemble.GradientBoostingClassifier()
gbc.fit(X, y)


# Get Feature Importance from the classifier
feature_importance = gbc.feature_importances_
# Normalize The Features
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(16, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center', color='#7A68A6')
plt.yticks(pos, np.asanyarray(df.columns[1:].tolist())[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig('Relative_Importance_GBoosting_5c.png')
plt.show()

"""--------------- selection de variables adaboost -------------- """

gbc = ensemble.AdaBoostClassifier()
gbc.fit(X, y)
# Get Feature Importance from the classifier
feature_importance = gbc.feature_importances_
# Normalize The Features
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(16, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center', color='#7A68A6')
plt.yticks(pos, np.asanyarray(df.columns[1:].tolist())[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig('Relative_Importance_AdaBoost_5c.png')
plt.show()

"""--------------- selection de variables adaboost -------------- """

gbc = ensemble.RandomForestClassifier()

gbc.fit(X, y)


# Get Feature Importance from the classifier
feature_importance = gbc.feature_importances_
# Normalize The Features
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(16, 12))
plt.barh(pos, feature_importance[sorted_idx], align='center', color='#7A68A6')
plt.yticks(pos, np.asanyarray(df.columns[1:].tolist())[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig('Relative_Importance_RandomForestClassifier.png')
plt.show()

#==============================================================================
"""---------------------matrix de confusion-------------------------------- """
#==============================================================================

pass_agg_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, linear_model.PassiveAggressiveClassifier))
grad_ens_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, ensemble.GradientBoostingClassifier))
decision_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, tree.DecisionTreeClassifier))
ridge_clf_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, linear_model.RidgeClassifier))
svm_svc_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, svm.SVC))
random_forest_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, ensemble.RandomForestClassifier))
k_neighbors_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, neighbors.KNeighborsClassifier))
logistic_reg_conf_matrix = metrics.confusion_matrix(y, stratified_cv(X, y, linear_model.LogisticRegression))
#dumb_conf_matrix = metrics.confusion_matrix(y, [0 for ii in y.tolist()]);  # ignore the warning as they are all 0

conf_matrix = {
                1: {
                    'matrix': pass_agg_conf_matrix,
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
plt.savefig('Confusion_Matrix_VC_5c.png')
plt.show()

    
#==============================================================================
"""------------------------ c4 == c5  ----------------------------""" 
#==============================================================================


g1=df[df.classe == "c1"] 
g2=df[df.classe == "c2"] 
g3=df[df.classe == "c3"] 
g4=df[df.classe == "c4"] 
g5=df[df.classe == "c5"] 
g6=df[df.classe == "c6"] 
try:
    g1["classe5"]="c1"
    g2["classe5"]="c2"
    g3["classe5"]="c3"
    g4["classe5"]="c4"
    g5["classe5"]="c4"
    g6["classe5"]="c6"
except:
   pass

newData =pd.concat([g1,g2,g3,g4,g5,g6],axis=0,ignore_index=True)

X = (newData[:][['moyenne','ecart-type','mediane','surface','entropie','uniformit','eccentricity']]).values
Y = (newData[:][['classe5']]).values
y = Y.ravel()

scaler= preprocessing.StandardScaler().fit(X)  
X = scaler.transform(X)

#==============================================================================
""" ----------------------- VotingClassifier--------------------------------"""
#==============================================================================
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = svm.SVC()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'SVM', 'Ensemble']):
      scores = cross_validation.cross_val_score(clf, X, y, cv=10, scoring='accuracy')
      print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


sklearn.ensemble()

