# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:49:45 2020

@author: tony
"""


import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Country column
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Male/Female
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier  = Sequential()

classifier.add(Dense(output_dim = 6 ,init = 'uniform' , activation='relu',input_dim=11))

classifier.add(Dense(output_dim = 6 ,init = 'uniform' , activation='relu'))

classifier.add(Dense(output_dim = 1 ,init = 'uniform' , activation='relu'))

classifier.compile(optimizer='adam' ,loss='binary_crossentropy' ,metrics=['accuracy'])

classifier.fit(X_train,y_train , batch_size=10,nb_epoch=10)

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)