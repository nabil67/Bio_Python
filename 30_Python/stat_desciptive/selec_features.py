# -*- coding: utf-8 -*-
"""
Created on Tue May 03 10:19:33 2016

@author: nabil.belahrach
"""

"""--------------- selection de variables -------------- """

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
plt.yticks(pos, np.asanyarray(df.columns.tolist())[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig('Relative_Importance_GBoosting.png')
plt.show()

