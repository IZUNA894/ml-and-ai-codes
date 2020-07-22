# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:19:27 2020

@author: tony
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

import  scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X ,method='ward',metric='euclidean'))
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('no of clusters')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5 , linkage ='ward' , affinity ='euclidean')
y_hc = hc.fit_predict(X)

#visulaizing the clusters
#plotting our clusters
plt.scatter(X[y_hc == 0 , 0],X[y_hc == 0 , 1] , s=1 , c='red' ,label = 'moderate')
plt.scatter(X[y_hc == 1 , 0],X[y_hc== 1 , 1] , s=5 , c='blue' ,label = 'careless')
plt.scatter(X[y_hc == 2 , 0],X[y_hc == 2 , 1] , s=5 , c='purple' ,label = 'target')
plt.scatter(X[y_hc == 3 , 0],X[y_hc == 3 , 1] , s=5 , c='yellow' ,label = 'sensible')
plt.scatter(X[y_hc == 4 , 0],X[y_hc == 4 , 1] , s=5 , c='cyan' ,label = 'rich but kanjoos')
plt.xlabel('salaries amount')
plt.ylabel('spending score')
plt.legend()
plt.show()