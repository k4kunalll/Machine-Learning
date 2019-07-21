import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
z = dataset.target

plt.scatter(X[z == 0, 0], X[z == 0, 1], c = "r" , label = "Setosa")
plt.scatter(X[z == 1, 0], X[z == 1, 1], c = "b" , label = "Versicolor")

plt.scatter(X[z == 2, 0], X[z == 2, 1], c = "g" , label = "Virginica")
plt.xlabel("SEPAL LENGTH")
plt.ylabel("SEPAL WIDTH")
plt.legend()
plt.title("ANALYSIS ON THE IRIS DATASET")
plt.show()



