import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)

svm.score(X_train, y_train)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.svm import SVC
svm = SVC()

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

from sklearn.ensemble import VotingClassifier
vote = VotingClassifier([ ("LogisticRegression", log_reg),
                          ("KNeighbour", knn),
                          ("NaiveBayes", nb),
                          ("SVM", svm),
                          ("DecisionTree", dt)
                          ])
        
vote.fit(X_train, y_train)
vote.score(X_train, y_train)

from sklearn.ensemble import BaggingClassifier



bag = BaggingClassifier(knn, n_estimators = 5)
bag.fit(X_train, y_train)
bag.score(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc.score(X_train, y_train)

predict = rfc.predict(X_test)










                            



