import pandas as pd
import numpy as np
import sklearn.model_selection as cv
import sklearn
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot
from IPython.display import Image
from statsmodels.stats.proportion import proportion_confint

def preprocess_for_decision_trees(dataset):
    # TODO
    return dataset

def execute_decision_trees(dataset):
    dataset = preprocess_for_decision_trees(dataset)

    # print(dataset.head())

    # X = dataset[['diag_1','diag_2','diag_3']]
    X = dataset[['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide']]
    # X = dataset.drop(['readmitted'], axis=1)
    y = dataset['readmitted']

    # print(X.shape)
    # print(X.head())

    (X_train, X_test, y_train, y_test) = cv.train_test_split(X, y, test_size=.3, random_state=0)

    clf = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = sklearn.metrics.accuracy_score(y_test, pred)

    print("Confusion matrix:")
    print(sklearn.metrics.confusion_matrix(y_test, pred))
    print()
    print("Accuracy on test set: ", score)
    print()
    print(sklearn.metrics.classification_report(y_test, pred))
    print("Confidence interval: ", proportion_confint(count=score*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test'))

    # dot_data = StringIO()
    # tree.export_graphviz(clf, out_file=dot_data,
    #                     filled=True, rounded=True,
    #                     feature_names=list(X.columns.values),
    #                     special_characters=True)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # Image(graph[0].create_png())