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

dataset.isnull().sum()

temp = pd.DataFrame(X[ : ,[1, 6, 13]])
temp[0].value_counts()
temp[1].value_counts()
temp[2].value_counts()

temp[0] = temp[0].fillna("Private")
temp[1] = temp[1].fillna("Prof-specially")
temp[2] = temp[2].fillna("United-States")

X[ : , [1, 6, 13]] = temp

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

lab.classes_

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features= [1,3,5,6,7,8,9,13])
X = one.fit_transform(X)
X = X.toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)



