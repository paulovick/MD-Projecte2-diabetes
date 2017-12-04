
# coding: utf-8

# # Meta Methods applied to the ionosphere data set

# In[1]:
import numpy as np    # Numeric and matrix computation
import pandas as pd   # Optional: good package for manipulating data 
import sklearn as sk  # Package with learning algorithms implemented

# Import libraries 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
#from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
import sklearn
from sklearn.metrics import precision_recall_fscore_support
import time

def bagging(ds):
	
	# utilitzarem totes les variables menys la que volem predir per a construir un classificador
	X=ds.drop(['readmitted'], axis=1)
	# volem predir la columna readmitted, aixi que sera la nostre y
	y=ds['readmitted']
	
	# separem el dataset en 70% training i 30% test
	(X_train, X_test,  y_train, y_test) = sklearn.model_selection.train_test_split(X, y, test_size=.3, random_state=1)


	# ## Voting scheme

	# In[3]:

	cv=10

	# ## Bagging

	# In[6]:

	nestValues = [1,2,5,10,20,50,100,200]
	featuresValues = [0.175, 0.350, 0.525, 0.700, 0.875]
	
	scoreT = []
	max_score = 0

	for nest in nestValues:
		for max_features in featuresValues:
			#Cross Validation
			ini = time.time()
			clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=nest,max_features=max_features)
			y_predicted = cross_val_predict(clf, X_train, y_train, cv=cv)
			end = time.time()
			print ("---------")
			print (str(nest) + ' Time spent = ' + str(end-ini) + 's')
			precision, recall, fscore, _ = precision_recall_fscore_support(y_train, y_predicted)
			print(precision, recall, fscore)
			p = (precision[0]+precision[1])/2.0
			r = (recall[0]+recall[1])/2.0
			f = (fscore[0]+fscore[1])/2.0
			m = (p+r)/2.0
			print(m)
			scoreT.append(m)
			if (m > max_score):
				max_score = m
				max_scorei = [nest, max_features]
			#print (sklearn.metrics.confusion_matrix(ytrain, ypredicted))
			#print (sklearn.metrics.classification_report(ytrain, ypredicted))
			print ("---------")

	print(str(max_score) + " " + str(max_scorei))
	
	
def adaboost(ds):
	# ## Boosting
	
	# utilitzarem totes les variables menys la que volem predir per a construir un classificador
	X=ds.drop(['readmitted'], axis=1)
	# volem predir la columna readmitted, aixi que sera la nostre y
	y=ds['readmitted']
	
	# separem el dataset en 70% training i 30% test
	(X_train, X_test,  y_train, y_test) = sklearn.model_selection.train_test_split(X, y, test_size=.3, random_state=1)


	# ## Voting scheme

	# In[3]:

	cv=10
	
	nestValues = [1,2,5,10,20,50,100,200]
	featuresValues = [0.175, 0.350, 0.525, 0.700, 0.875]
	
	scoreT = []
	max_score = 0
	
	# In[8]:
	
	# In[9]:
	
	for nest in nestValues:
		#Cross Validation
		ini = time.time()
		clf = BaggingClassifier(base_estimator=AdaBoostClassifier(DecisionTreeClassifier(max_depth=5)),n_estimators=nest)
		y_predicted = cross_val_predict(clf, X_train, y_train, cv=cv)
		end = time.time()
		print ("---------")
		print (str(nest) + ' Time spent = ' + str(end-ini) + 's')
		precision, recall, fscore, _ = precision_recall_fscore_support(y_train, y_predicted)
		print(precision, recall, fscore)
		p = (precision[0]+precision[1])/2.0
		r = (recall[0]+recall[1])/2.0
		f = (fscore[0]+fscore[1])/2.0
		m = (p+r)/2.0
		print(m)
		scoreT.append(m)
		if (m > max_score):
			max_score = m
			max_scorei = [nest]
		#print (sklearn.metrics.confusion_matrix(ytrain, ypredicted))
		#print (sklearn.metrics.classification_report(ytrain, ypredicted))
		print ("---------")

	print(str(max_score) + " " + str(max_scorei))
