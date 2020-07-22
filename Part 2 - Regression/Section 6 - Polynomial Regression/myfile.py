# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:33:25 2020

@author: tony
"""


# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2].values

#fitting linear model in datashet
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)
y_pred = regressor.predict(X)
print(regressor.predict([[6.5]]))

#fitting polynomial regression into dataset
from sklearn.preprocessing import PolynomialFeatures


#for degree 2
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
#poly_reg.fit(X_poly,y)
regressor2 = LinearRegression()
regressor2.fit(X_poly,y)
y_pred2 = regressor2.predict(X_poly)

print(regressor2.predict(poly_reg.fit_transform([[6.5]])))


#for degree 3
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
#poly_reg.fit(X_poly,y)
regressor2 = LinearRegression()
regressor2.fit(X_poly,y)
y_pred3 = regressor2.predict(X_poly)
print(regressor2.predict(poly_reg.fit_transform([[6.5]])))


#for degree =4
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
#poly_reg.fit(X_poly,y)
regressor2 = LinearRegression()
regressor2.fit(X_poly,y)
y_pred4 = regressor2.predict(X_poly)
print(regressor2.predict(poly_reg.fit_transform([[6.5]])))


#visualization of data from linear model
plt.scatter(X,y, color = 'red')
plt.plot(X,y_pred,color='blue')
plt.plot(X,y_pred2,color='yellow')
plt.plot(X,y_pred3 , color ='pink')
plt.plot(X,y_pred4,color = 'purple')
plt.show()

#predicting the answer for exp 6.5



