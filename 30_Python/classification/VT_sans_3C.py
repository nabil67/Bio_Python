# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 11:41:33 2016

@author: nabil.belahrach@veolia.com
"""

# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA




data = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/20_brut/VTsansVesicule.csv", 
                   sep=";", header = False, names = ['classe','moyenne','mediane','surface'],
                   dtype={"classe":object, "moyenne":object, "mediane":float, "surface":float})
                   
print data.head()                   
data.shape

#==============================================================================
"""-----------------Préparation des données ------------------ """
#==============================================================================

data['moyenne']=data['moyenne'].str.replace("," , ".")
data["classe"] =pd.Categorical(data["classe"])
data[['moyenne','mediane','surface']] = data[['moyenne','mediane','surface']].astype(float)
classe=pd.CategoricalIndex(data["classe"]).categories
data["classe"] = data["classe"].cat.rename_categories(["c1","c2","c3","c6","c4","c5"])

#data.to_csv('myFile.csv', sep = ';')

#==============================================================================
"""--------------------inventaire des classes-----------------"""  
#==============================================================================
print pd.value_counts(data["classe"], sort=False)
  
#------Plot a histogram of frequencies
data.classe.value_counts().plot(kind='barh')
plt.title('nombre des apparences des classe')
plt.xlabel('Frequency')

#-----Now make a pie chart
data.classe.value_counts().plot(kind='pie')
plt.axis('equal')
plt.title('Les six classes ')

#==============================================================================
""" On fusionne les classes k1<- c1+c2+c3 d'une part et  k2 <- c4  , k3 <- c5+c6  """
#==============================================================================

X = data[:][['moyenne', 'mediane', 'surface']].values

k1 = sum(data.classe == "c1") + sum(data.classe == "c2")+ sum(data.classe =="c3")

k3 = sum( data.classe == "c4") + sum( data.classe == "c5") + sum(data.classe =="c6")

tmp1 = [0]
tmp1=np.repeat(tmp1, k1).astype(int)
tmp2 = [1]
tmp2=np.repeat(tmp2, k2).astype(int)
tmp3 = [2]
tmp3=np.repeat(tmp3, k3).astype(int)

Y = np.hstack((tmp1,tmp2,tmp3)).ravel()
#==============================================================================
""" ------------------------ Classification KNN --------------------------- """
#==============================================================================
""" ------center et réduire les variables explicatives ! -----------"""
scaler= preprocessing.StandardScaler().fit(X)  
X = scaler.transform(X)

X_train,X_test,Y_train,Y_test = train_test_split( X, Y, 
                                                 test_size= 0.4,random_state=33)

               # entrainement 
param = [{"n_neighbors": list(range(1,15))}]
knn = GridSearchCV(KNeighborsClassifier(), param, cv=10, n_jobs= -1)  # 5-fold cv
digit_knn = knn.fit(X_train, Y_train)  
print ("le best param = "), digit_knn.best_params_["n_neighbors"]                              # best param = 9

""" ------on relance le modèle avec le best-paramètre -----------"""

knn = KNeighborsClassifier(n_neighbors= digit_knn.best_params_["n_neighbors"])

digit_knn.score(X_train,Y_train)     # estimation de l'erreur = 55%
Y_pred = digit_knn.predict(X_test)   # prediction des réponses de X_test
table = pd.crosstab( Y_test, Y_pred) # matrice de confusion
print table;
print classification_report( Y_test, Y_pred)
plt.matshow(table)
plt.title("Matrice de Confusion du knn")   # pas top
plt.colorbar()
plt.show()

#==============================================================================
"""-------------------------- visualisation --------------------------- """
#==============================================================================

fig = plt.figure()
ax = fig.add_subplot(111, projection= '3d')

x = X[:, 0].astype(float)
y = X[:, 1].astype(float)
z = X[:, 2].astype(float)

df = np.vstack((Y,X[:,0], X[:,1], X[:,2])).T

fig = plt.figure(1)
ax = Axes3D(fig)

ax.scatter(X[:,0] , X[:,1], X[:,2], c=df[:,0])           
ax.set_title(" visualusation 3D des classes")
ax.set_xlabel("la moyenne")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("la mediane")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("la surface des pixels")
ax.w_zaxis.set_ticklabels([])
ax.legend()
plt.show()
#==============================================================================
""" ------------TODO  ! ---------- svm multi-class, plutard ! -----------------------------"""
#==============================================================================

clf = svm.SVC(kernel ='rbf', cv = 10)
digit_svm = clf.fit(X_train, Y_train)

clf.decision_function_shape = "ovr"
Y_pred = clf.predict(X_test)
pd.crosstab(Y_test, Y_pred)
print classification_report( Y_test, Y_pred)

#==============================================================================
""" -------TODO ! -----visualisation sous composantes acp----------------- """
#==============================================================================
X_reduced = PCA(n_components=3).fit_transform(X)
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,
           cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()

#==============================================================================
"""---------------------- bayésien naif -------------------------------  """
#==============================================================================
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
Y_pred = gnb.fit(X_train, Y_train).predict(X_test)
table = pd.crosstab( Y_test, Y_pred) 
plt.matshow(table)
classes = ['classe_1','classe_2','classe_3']
print (classification_report(Y_test, Y_pred, target_names = classes))
gnb.score(X_train, Y_train)

#==============================================================================
"""---------------------- Regression logistique -------------------------"""
#==============================================================================
from sklearn.linear_model import  LogisticRegressionCV

logm = LogisticRegressionCV( cv=10)

clf_logm=logm.fit(X_train, Y_train)
Y_pred = clf_logm.predict(X_test)
table = pd.crosstab(Y_test, Y_pred)
print table
plt.matshow(table)
plt.title("logit_M_Confusion_knn 3c")   
plt.colorbar()
print (classification_report(Y_test, Y_pred, target_names = ['0','1','2']))

#==============================================================================
"""-------------------------- KMeans-------------------------"""
#==============================================================================
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

colors = ["g.","r.","c.","y."]
fig = plt.figure(1)
ax = Axes3D(fig)
for i in range(len(X)):
    #print("coordinate:",X[i], "label:", labels[i])
    ax.scatter(X[i][0], X[i][1],X[i][2], colors[labels[i]] );


ax.scatter(centroids[:, 0],centroids[:, 1],centroids[:, 2],
            marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()



