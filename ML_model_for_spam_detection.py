import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import mean_squared_error 

data_set = pd.read_csv("spam_data_set.csv")

featuters = data_set['EmailText'].values
target = data_set['Label'].values

featuters_train = featuters[0 : 4458]
target_train = target[0 : 4458]

featuters_test = featuters[4458 : ] 
target_test = target[4458 : ]

cv = CountVectorizer()
metrix_train = cv.fit_transform(featuters_train)
metrix_test = cv.transform(featuters_test)

model = MultinomialNB()
model.fit(metrix_train,target_train)

print(model.predict(metrix_test))

