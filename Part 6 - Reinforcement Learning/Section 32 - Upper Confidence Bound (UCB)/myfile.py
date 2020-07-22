# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:40:16 2020

@author: tony
"""


import pandas as pd
import matplotlib.pyplot as plt
   
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing UCB
import math
N = 10000
d = 10
ads_selected = []
numberOfSelections = [0]*d
sumOfRewards = [0]*d
totalRewards = 0

for n in range(0,N):
    ad = 0 
    maxUpperBound = 0
    for i in range(0,d):
        if(numberOfSelections[i]>0):
            avgRewards = sumOfRewards[i] /numberOfSelections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numberOfSelections[i] )
            upperBound = avgRewards + delta_i
        else:
            upperBound = 1e400
        
        if(upperBound > maxUpperBound):
            maxUpperBound = upperBound
            ad = i
    ads_selected.append(ad)
    numberOfSelections[ad] = numberOfSelections[ad] + 1
    sumOfRewards[ad] = sumOfRewards[ad] + dataset.values[n,ad]
    totalRewards = totalRewards + dataset.values[n,ad]
    
#plotting a graph
plt.hist(ads_selected)
plt.title('upper confidence bound')
plt.xlabel('ads version')
plt.ylabel('times they were selected')