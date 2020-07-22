# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:32:38 2020

@author: tony
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters = i , init = 'k-means++' ,n_init = 10 ,max_iter = 300 ,random_state = 0 )
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)

plt.plot(range(1,11) , wcss)
plt.ylabel('wcss value')
plt.xlabel('no. of clusters')
plt.title('elbow method')
plt.legend()
plt.show()

#making clusters
Kmeans = KMeans(n_clusters = 5 , init = 'k-means++' ,n_init = 10 ,max_iter = 300 ,random_state = 0 )
y_Kmeans = Kmeans.fit_predict(X)
#plotting our clusters
plt.scatter(X[y_Kmeans == 0 , 0],X[y_Kmeans == 0 , 1] , s=1 , c='red' ,label = 'moderate')
plt.scatter(X[y_Kmeans == 1 , 0],X[y_Kmeans == 1 , 1] , s=5 , c='blue' ,label = 'careless')
plt.scatter(X[y_Kmeans == 2 , 0],X[y_Kmeans == 2 , 1] , s=5 , c='purple' ,label = 'target')
plt.scatter(X[y_Kmeans == 3 , 0],X[y_Kmeans == 3 , 1] , s=5 , c='yellow' ,label = 'sensible')
plt.scatter(X[y_Kmeans == 4 , 0],X[y_Kmeans == 4 , 1] , s=5 , c='cyan' ,label = 'rich but kanjoos')
plt.scatter(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1] , s=100 ,c='green' , label= 'centroid')
plt.xlabel('salaries amount')
plt.ylabel('spending score')
plt.legend()
plt.show()