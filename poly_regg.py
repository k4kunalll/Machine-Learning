import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

m = 100
X = 8 * np.random.randn(m ,1)
y = 0.6 * X ** 2 + 4 * X + 2 +np.random.randn(m ,1)

plt.scatter(X, y)
plt.axis([-10, 10, -70, 200 ])
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2, include_bias = False)
X_poly = poly.fit_transform(X)


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

X_new = np.linspace(-3 ,3, 100).reshape(-1, 1)
X_new_poly = poly.fit_transform(X_new)
y_new = lin_reg.predict(X_new_poly)

plt.scatter(X, y)
plt.plot(X_new, y_new , c ="r")
plt.axis([-2.0 ,5.0 ,-70, 200 ])
plt.show()
