import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

plt.scatter(X[y == 0, 0], X[y == 0, 1], c = "r" , label = "Setosa")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c = "b" , label = "Versicolor")
plt.scatter(X[y == 2, 0], X[y == 2, 1], c = "g" , label = "Virginica")
plt.xlabel("SEPAL LENGTH")
plt.ylabel("SEPAL WIDTH")
plt.legend()
plt.title("ANALYSIS ON THE IRIS DATASET")
plt.show()



