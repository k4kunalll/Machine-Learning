import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_boston

dataset = load_boston()
X = dataset.data
y = dataset.target

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

score = lin_reg.score(X, y)
coefficient = lin_reg.coef_
intercept = lin_reg.intercept_

y_pred = lin_reg.predict([[11,12,58,45,44,16,35,14,44,45,78,95,44]])


