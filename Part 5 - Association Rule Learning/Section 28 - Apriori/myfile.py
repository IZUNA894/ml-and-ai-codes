# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 00:56:15 2020

@author: tony
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing dataset 
dataset = pd.read_csv('Market_Basket_Optimisation.csv',header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20) ])
    
#training dataset
from apyori import apriori
rules = apriori(transactions ,min_support = 0.3 , min_confidence = 0.2,min_lift = 3 ,min_length = 2 )

#visualization
results = list(rules)