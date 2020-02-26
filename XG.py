import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("churn_Modelling.csv")

X = dataset.iloc[ : , 3:13].values
y = dataset.iloc[ : , -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()

X[ : , 1] = le.fit_transform(X[ : ,1])
X[ : , 2] = le.fit_transform(X[ : ,2])

ohe = OneHotEncoder(categorical_features = [1])

X = ohe.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from xgboost import XGBClassifier 
classi = XGBClassifier()
classi.fit(X_train, y_train) 

y_pred = classi.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)