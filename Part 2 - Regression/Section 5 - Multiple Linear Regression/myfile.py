# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:30:29 2020

@author: tony
"""


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing databases
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4:].values

#encoding catergorial  data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder' , OneHotEncoder(),[3])],remainder="passthrough")
X = np.array(ct.fit_transform(X), dtype=np.float)

#avoiding dummy variable trap...
X =X[:,1:]

#seprating train and test models
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)

#fitting data into multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting data
y_pred = regressor.predict(X_test)


#building the optimal model using the backward elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values = X ,axis = 1)
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
print (regressor_OLS.summary())

X_ne1 = X_test[:,[0,3]]
y_pred2 = regressor_OLS.predict(X_ne1)








