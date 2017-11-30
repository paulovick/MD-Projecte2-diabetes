import pandas as pd
import numpy as np
import sklearn.model_selection as cv
import sklearn

def decisionTree(ds):
	ds.head()
	## Separate data from labels
	X=ds.drop(['readmitted'], axis=1)
	y=ds['readmitted']

	print(X.shape)
	X.head()

	## Transform to numerical dataset
	Xn=pd.get_dummies(X)
	Xn.head()

	## Split into training and tesr
	(X_train, X_test,  y_train, y_test) = cv.train_test_split(Xn, y, test_size=.3, random_state=1)

	from sklearn.naive_bayes import BernoulliNB  # For binari features (f.i. word appears or not in document)
	from statsmodels.stats.proportion import proportion_confint
	
	from sklearn.naive_bayes import GaussianNB  # For numerical featuresm assuming normal distribution
	from sklearn.naive_bayes import MultinomialNB  # For features with counting numbers (f.i. hown many times word appears in doc)
	from sklearn.naive_bayes import BernoulliNB  # For binari features (f.i. word appears or not in document)
	from sklearn.metrics import confusion_matrix
	# interval confidence
	from statsmodels.stats.proportion import proportion_confint
	# No parameters to tune

	clf = GaussianNB()
	pred = clf.fit(X_train, y_train).predict(X_test)
	print(confusion_matrix(y_test, pred))
	print()
	print("Accuracy:", sklearn.metrics.accuracy_score(y_test, pred))
	print()
	print(sklearn.metrics.classification_report(y_test, pred))
	epsilon = sklearn.metrics.accuracy_score(y_test, pred)
	proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test')

	from sklearn import tree
	from sklearn.externals.six import StringIO  
	import pydot
	#from IPython.display import Image  
	
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	pred = clf.predict(X_test)

	# Obtain accuracy score of learned classifier on test data
	print(clf.score(X_test, y_test))
	print(confusion_matrix(y_test, pred))
	print()
	print("Accuracy:", sklearn.metrics.accuracy_score(y_test, pred))
	print()
	print(sklearn.metrics.classification_report(y_test, pred))
	epsilon = sklearn.metrics.accuracy_score(y_test, pred)
	proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test')
	
	clf=tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=2,min_impurity_split=0.2)
	clf = clf.fit(X_train, y_train)
	pred = clf.predict(X_test)

	# Obtain accuracy score of learned classifier on test data
	print(clf.score(X_test, y_test))
	print(confusion_matrix(y_test, pred))
	print()
	print("Accuracy:", sklearn.metrics.accuracy_score(y_test, pred))
	print()
	print(sklearn.metrics.classification_report(y_test, pred))
	epsilon = sklearn.metrics.accuracy_score(y_test, pred)
	print("Interval of confudence:", proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test'))

	dot_data = StringIO() 
	tree.export_graphviz(clf, out_file=dot_data,    
							 filled=True, rounded=True, 
							 special_characters=True)  
	graph = pydot.graph_from_dot_data(dot_data.getvalue())  
	graph[0].write_png('DT-plots/DT1.png')
	print("imatge guardada a DT-plots/DT1.png")
	#
	# imatge massa gran per a les llibreries, fara crashejar python
	#
	#Image(graph[0].create_png())
	#import cv2 
	#input = cv2.imread('DT-plots/DT1.png')
	#input = cv2.resize(input, (600, 100))    
	#cv2.imshow('DT plot', input)
	#cv2.waitKey(5000)
	#cv2.destroyAllWindows()
	from sklearn.model_selection import GridSearchCV
	params = {'min_impurity_decrease': list(np.linspace(0,1,21)),'min_samples_split':list(range(2,102,11))}
	clf = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy'), param_grid=params,cv=10,n_jobs=-1)  # If cv is integer, by default is Stratifyed 
	clf.fit(X_train, y_train)
	print("Best Params=",clf.best_params_, "Accuracy=", clf.best_score_)
	
	clf=tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=2,min_impurity_decrease=0.2)
	clf = clf.fit(X_train, y_train)
	pred = clf.predict(X_test)

	# Obtain accuracy score of learned classifier on test data
	print(clf.score(X_test, y_test))
	print(confusion_matrix(y_test, pred))
	print()
	print("Accuracy:", sklearn.metrics.accuracy_score(y_test, pred))
	print()
	print(sklearn.metrics.classification_report(y_test, pred))
	epsilon = sklearn.metrics.accuracy_score(y_test, pred)
	proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test')
	
	dot_data = StringIO() 
	tree.export_graphviz(clf, out_file=dot_data,    
							 filled=True, rounded=True, 
							 special_characters=True)  
	graph = pydot.graph_from_dot_data(dot_data.getvalue())  
	graph[0].write_png('DT-plots/DT2.png')
	print("imatge guardada a DT-plots/DT2.png")
	#Image(graph[0].create_png())  
		
