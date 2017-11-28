import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from pandas.tools import plotting
from preprocessing import *
from decision_trees import *

dataset = read_and_filter_dataset(preprocess=True,nrows=5000,save_csv=True)
# dataset = read_and_filter_dataset(use_preprocessed=True)

# plot_statistics(dataset)
# plot_null_statistics(dataset)

execute_decision_trees(dataset)


# X = np.array(dataset.drop('readmitted',1))
# X = np.array(dataset[['gender','race','age']])
# y = np.array(dataset['readmitted'])

# from sklearn import cross_validation
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
# from os import system

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
# print(X_train.shape)
# print(X_test.shape)

# clf = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
# clf.score(X_test, y_test)     
# y_pred = clf.predict(X_test)
# print("Classification Report:")
# print(metrics.classification_report(y_test, y_pred))
# print("Confusion Matrix:")
# print(metrics.confusion_matrix(y_test, y_pred))