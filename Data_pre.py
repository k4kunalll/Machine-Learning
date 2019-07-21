import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("S:\Jupyter notebook\M.L Summer\Data_Pre.csv")
X = dataset.iloc[ : , 0:3].values
y = dataset.iloc[ : ,-1].values

from sklearn.impute import SimpleImputer
sim = SimpleImputer()
X[ : , 0:2] = sim.fit_transform(X[ : , 0:2])

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
lab.fit(X[ : , 2])
X[ : , 2] = lab.transform(X[ : , 2])
lab.classes_
lab.fit(y)
y = lab.transform(y)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features=[2])
X = one.fit_transform(X)
X= X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

 
