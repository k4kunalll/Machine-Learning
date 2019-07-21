import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import re

nltk.download('stopwords')

dataset = pd.read_csv("train.csv")
dataset["tweet"][0]

processed_tweet = []

for i in range(31962):
    tweet = re.sub('@[\w]*' ,' ', dataset['tweet'][i])
    tweet = re.sub('[^a-zA-Z#]',' ',tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    
    tweet = [ps.stem(token) for token in tweet if not token in stopwords.words('english')]
    tweet = ' '.join(tweet)
    processed_tweet.append(tweet)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X = cv.fit_transform(processed_tweet)
X = X.toarray()
y = dataset["label"].values

from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()
n_b.fit(X, y)

print(cv.get_feature_names())