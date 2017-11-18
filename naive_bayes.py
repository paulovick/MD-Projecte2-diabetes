import numpy as np                     # Llibreria matemÃ tica
import matplotlib.pyplot as plt        # Per mostrar plots
import sklearn                         # Llibreia de DM
import sklearn.datasets as ds            # Per carregar mÃ©s facilment el dataset digits
import sklearn.model_selection as cv    # Pel Cross-validation
import sklearn.neighbors as nb           # Per fer servir el knn
# Obtain Recall, Precision and F-Measure for each class
from sklearn import metrics

# interval confidence
from sklearn.cross_validation import cross_val_score
from statsmodels.stats.proportion import proportion_confint

from sklearn.naive_bayes import GaussianNB  # For numerical featuresm assuming normal distribution
from sklearn.naive_bayes import MultinomialNB  # For features with counting numbers (f.i. hown many times word appears in doc)
from sklearn.naive_bayes import BernoulliNB  # For binari features (f.i. word appears or not in document)


def naive_bayes(df):
    # Separate data from labels

    y_name = "readmitted" # y value to predict
    y = df[y_name].values
    i, = np.where(df.columns.values == y_name) # index column of y value
    # IMPORTANT: nomes poden ser variables numeriques o categoriques transformades en numeriques?
    X = df.values[:,[1,6,7,8]]#np.delete(df.values, i, 1) # Delete y value column
    print (X)
    # Let's do a simple cross-validation: split data into training and test sets (test 30% of data)
    (X_train, X_test,  y_train, y_test) = cv.train_test_split(X, y, test_size=.3, random_state=1)

    # No parameters to tune
    vc=50
    clf = GaussianNB()
    scores = cross_val_score(clf, X, y, cv=vc, scoring='accuracy')
    print("Accuracy: %0.3f" % (scores.mean()))


    pred = clf.fit(X_train, y_train).predict(X_test)
    print(sklearn.metrics.confusion_matrix(y_test, pred))
    print()
    print("Accuracy:", sklearn.metrics.accuracy_score(y_test, pred))
    print()
    print(metrics.classification_report(y_test, pred))
    epsilon = sklearn.metrics.accuracy_score(y_test, pred)
    exit(1)
    proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test')