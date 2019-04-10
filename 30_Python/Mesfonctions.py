# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:54:03 2016

@author: nabil.belahrach
"""
import numpy as np
from sklearn import cross_validation



def Acccuracy3diag(table):
    mat = table.values
    bc = 0
    for i in [-1,0,1]:        
        diag= np.trace(mat,offset= i )
        bc += diag 
    total = sum(sum(mat[:,:]))
    prc = float(bc)/total
    return prc
    
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
    

if __name__=='__main__':
    Acccuracy3diag()
    stratified_cv()
    