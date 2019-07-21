import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("S:\Jupyter notebook\M.L Summer\Adult_Data.csv" ,names = ["age",
                                                                                "workclass",
                                                                                "fnlwgt",
                                                                                "education",
                                                                                "education-num",
                                                                                "marital-status",
                                                                                "occupation",
                                                                                "relationship",
                                                                                "race",
                                                                                "sex",
                                                                                "capital_gain",
                                                                                "capital_loss",
                                                                                "hours_per_week",
                                                                                "native-country",
                                                                                "Salary"])

X = dataset.iloc[ : , 0:14].values
y = dataset.iloc[ : ,-1].values

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
X[ : ,1] = lab.fit_transform(X[ : , 1])
X[ : ,3] = lab.fit_transform(X[ : , 3])
X[ : ,5] = lab.fit_transform(X[ : , 5])
X[ : ,6] = lab.fit_transform(X[ : , 6])
X[ : ,7] = lab.fit_transform(X[ : , 7])
X[ : ,8] = lab.fit_transform(X[ : , 8])
X[ : ,9] = lab.fit_transform(X[ : , 9])
X[ : ,13] = lab.fit_transform(X[ : , 13])
y = lab.fit_transform(y)

from sklearn.impute import SimpleImputer
sim = SimpleImputer(missing_values = 0, strategy = "median")
X[ : , 0:9] = sim.fit_transform(X[ : , 0:9])
X[ : , 13:14] = sim.fit_transform(X[ : , 13:14])

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features= [1,3,5,6,7,8,9,13])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)







