# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:51:22 2020

@author: tony
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:40:16 2020

@author: tony
"""


import pandas as pd
import matplotlib.pyplot as plt
   
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing UCB
import random
N = 10000
d = 10
ads_selected = []
totalRewards = 0
numberOfRewards0 = [0]*d
numberOfRewards1 = [0]*d

for n in range(0,N):
    ad = 0 
    maxRandom = 0
    for i in range(0,d):
        randomBeta = random.betavariate(numberOfRewards1[i] + 1 , numberOfRewards0[i] + 1)
        
        if(randomBeta > maxRandom):
            maxRandom = randomBeta
            ad = i
            
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    
    if(reward == 0):
        numberOfRewards0[ad] = numberOfRewards0[ad] + 1
    else:
        numberOfRewards1[ad] = numberOfRewards1[ad] + 1
        
    totalRewards = totalRewards + reward
    
#plotting a graph
plt.hist(ads_selected)
plt.title('upper confidence bound')
plt.xlabel('ads version')
plt.ylabel('times they were selected')