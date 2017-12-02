import numpy as np
import pandas as pd
import csv
import sklearn.model_selection as cv
import sklearn
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier


def adaboost(ds):
	# utilitzarem totes les variables menys la que volem predir per a construir un classificador
	X=ds.drop(['readmitted'], axis=1)
	# volem predir la columna readmitted, aixi que sera la nostre y
	y=ds['readmitted']
	
	# ho convertim en numeric dataset (encara que en teoria ja ho esta)
	Xn=pd.get_dummies(X)
	
	# separem el dataset en 70% training i 30% test
	(X_train, X_test,  y_train, y_test) = cv.train_test_split(Xn, y, test_size=.3, random_state=1)
	
	# adaboost amb decision stumps (recordem: els decision stumps son decision trees de altura 1)
	bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1))
	bdt = bdt.fit(X_train, y_train)
	resultat_decision_stumps = np.sum(bdt.predict(X_test) == y_test) / len(X_test)
	print("resultat de adaboost amb decision stumps: " + str(resultat_decision_stumps) + "\n")
	
	X_folds = np.array_split(X, 10)
	y_folds = np.array_split(y, 10)
	scores = list()
	for k in range(10):
		# We use 'list' to copy, in order to 'pop' later on
		X_tr = list(X_folds)
		X_te  = X_tr.pop(k)
		X_tr = np.concatenate(X_tr)
		y_tr = list(y_folds)
		y_te  = y_tr.pop(k)
		y_tr = np.concatenate(y_tr)
		scores.append(bdt.fit(X_tr, y_tr).score(X_te, y_te))
	print("K-fold cross validation score: " + str(np.mean(scores)) + " +- " + str(np.std(scores)) + "\n")
	
	value_max = None
	depth_max = None
	
	for d in range(1,21): # 21 per fer uns quants valors, he provat fins a 1000
		bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=d))
		bdt = bdt.fit(X_train, y_train)
		res = np.sum(bdt.predict(X_test) == y_test) / len(X_test)
		print("resultat de adaboost amb decision tree (depth=" + str(d) + "): " + str(res))
		if value_max is None or res > value_max:
			value_max = res
			depth_max = d
	print("\nmillor valor: {} \n amb profundidat {}".format(value_max, depth_max))
