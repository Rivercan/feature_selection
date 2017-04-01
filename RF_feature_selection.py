# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os, csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

project_path = os.path.dirname(os.path.abspath(__file__))
csv_data = project_path + "/Dataset/LargeTrain.csv"

data = np.genfromtxt(csv_data,delimiter=',')
col_row = data.shape

row = col_row[0]
col = col_row[1]

X= data[1:row,0:col-1]
Y= data[1:row,col-1:col]
y=Y.ravel()
y1=y.astype(int)

forest = RandomForestClassifier()
forest.fit(X, y1)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
cut_indices = indices[0:10]

header_indices = np.copy(cut_indices)
top10_indices = np.append(header_indices,[-1])

df = pd.read_csv(csv_data)
out = df.iloc[:,top10_indices]
out.to_csv(project_path + "/RF_top10.csv",index=False)

# Print the feature ranking
print("Feature ranking:")
for count in range(10):
    print("%d. no.%d (%f)" % (count+1, indices[count], importances[indices[count]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(10), importances[cut_indices],color="b", yerr=std[cut_indices], align="center")
plt.xticks(range(10), cut_indices)
plt.xlim([-1, 10])
plt.show()
