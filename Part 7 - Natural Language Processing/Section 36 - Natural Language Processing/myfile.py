# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:21:38 2020

@author: tony
"""

import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus =[]

nltk.download('stopwords')

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t' ,quoting = 3 )

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#making a word bag
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#making train and test sets
from sklearn.model_selection import train_test_split
X_Train,X_Test , y_Train , y_Test = train_test_split( X,y , test_size = 0.20 ,random_state = 0)


#making our classification model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_Train,y_Train)

#predicting the test result
y_pred = classifier.predict(X_Test)

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_Test, y_pred)
print(cm)