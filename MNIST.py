import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#from sklearn.datasets import load_digits
#dataset = load_digits()

from sklearn.datasets import fetch_mldata
dataset = fetch_mldata("MNIST original")


X = dataset.data
y = dataset.target

some_digit = X[47856]
some_digit_image = some_digit.reshape(28 ,28)

plt.imshow(some_digit_image)
plt.show()

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X, y)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=17)
dtc.fit(X ,y)

dtc.score(X, y)

dtc.predict(X[[47856, 25876, 25479,36945,58245], 0:784])
