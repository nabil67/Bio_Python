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

data = pd.read_csv("U:/Stagiaires/Nabil.BELAHRACH/Donnees/20_brut/VTsansVesicule.csv", 
                   sep=";", header = False, names = ['classe','moyenne','mediane','surface'],
                   dtype={"classe":object, "moyenne":object, "mediane":float, "surface":float})
                   
print data.head()                   
data.shape
str(data['moyenne'])

"""---------------remplacer ',' par '.' --------------"""

data['moyenne']=data['moyenne'].str.replace("," , ".")
data["classe"] =pd.Categorical(data["classe"], ordered=False)
data["classe"] = data["classe"].cat.rename_categories(["c1","c2","c3","c6","c4","c5"])

#==============================================================================
"""--------------------inventaire des classes-----------------"""  
#==============================================================================
   
cl=["c1","c2","c3","c4","c5","c6"]
desc=[]
for i in xrange(len(cl)):
    desc.append(sum(data["classe"] == cl[i]))
    print cl[i],"=", sum(data["classe"] == cl[i])
    
print pd.DataFrame([cl, desc])  

#------Plot a histogram of frequencies
data.classe.value_counts().plot(kind='barh')
plt.title('nombre des apparences des classe')
plt.xlabel('Frequency')

#-----Now make a pie chart
data.classe.value_counts().plot(kind='pie')
plt.axis('equal')
plt.title('Les classes ')

"""------- Transformation  dataframe_to_array --------"""

data[['moyenne', 'mediane', 'surface']] = data[['moyenne', 'mediane', 'surface']].astype(float)

 Y = data[:][['classe']].values 
 X = data[:][['moyenne', 'mediane', 'surface']].values
 Y =  np.ravel(Y)
""" -------split the data into a train and a test (25%) set -------"""

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size= 0.25, random_state=33)

""" ------center et réduire les variables explicatives ! -----------"""

scaler= preprocessing.StandardScaler().fit(X_train)   # pas besoin pour les knn !


Y_train =  np.ravel(Y_train)                           # conversion 
Y_test =  np.ravel(Y_test)
digit_knn = knn.fit(X_train, Y_train)                 # entrainement 
param = [{"n_neighbors": list(range(1,15))}]
knn = GridSearchCV(KNeighborsClassifier(), param, cv=6, n_jobs=1)  # 5-fold cv
digit_knn.best_params_["n_neighbors"]                              # best param = 9

""" ------on relance le modèle avec le best-paramètre -----------"""

knn = KNeighborsClassifier(n_neighbors= digit_knn.best_params_["n_neighbors"])

digit_knn.score(X_train,Y_train)     # estimation de l'erreur = 55%
Y_pred = digit_knn.predict(X_test)   # prediction des réponses de X_test
table = pd.crosstab( Y_test, Y_pred) # matrice de confusion
print table;
plt.matshow(table)
plt.title("Matrice de Confusion du knn 6c")   # pas top
plt.colorbar()
plt.show()


#==============================================================================
"""-------------   ----------------------  """
#==============================================================================
 

