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
    y_name = "readmitted"  # y value to predict
    y = df[y_name].values
    #i, = np.where(df.columns.values == y_name)  # index column of y value

    X_bernoulli = bernoulli(y_name, df)
    X_multinomial, i_multinomial = multinomial(df)
    X_gaussian = gaussian(df, i_multinomial)
    X_array = (("Bernoulli", X_bernoulli), ("Multinomial", X_multinomial), ("Gaussian", X_gaussian))

    for method, X in X_array:
        # Let's do a simple cross-validation: split data into training and test sets (test 30% of data)
        (X_train, X_test, y_train, y_test) = cv.train_test_split(X, y, test_size=.3, random_state=1)

        # No parameters to tune
        vc = 50
        clf = BernoulliNB()
        scores = cross_val_score(clf, X, y, cv=vc, scoring='accuracy')
        print("%s Accuracy ( %s variables): %0.3f" % (method, len(X[0]), scores.mean()))

        # pred = clf.fit(X_train, y_train).predict(X_test)
        # print(sklearn.metrics.confusion_matrix(y_test, pred))
        # print()
        # print("%s acurracy:" % (method), sklearn.metrics.accuracy_score(y_test, pred))
        # print()
        # print(metrics.classification_report(y_test, pred))
        # epsilon = sklearn.metrics.accuracy_score(y_test, pred)
        # proportion_confint(count=epsilon * X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test')

def bernoulli(y_name, df):
    # Binaries
    nom_binaries = [col for col in df if df[col].nunique() == 2]
    nom_binaries.remove(y_name)
    i_binaries = [np.where(df.columns.values == b)[0][0] for b in nom_binaries]
    X = df.values[:, i_binaries]
    return X

def multinomial(df):
    i_multinomial = [7, 9, 10, 11, 12, 13, 14, 17]
    X = df.values[:, i_multinomial]
    return X, i_multinomial

def gaussian(df, i_multinomial):
    nom_gaussianes = [col for col in df if df[col].nunique() > 2]
    i_nobinaries = [np.where(df.columns.values == b)[0][0] for b in nom_gaussianes]
    i_nobinaries = list(set(i_nobinaries) - set(i_multinomial))
    remove_correlations = ["admission_type_id"]  # y value to predict
    i_correlations = [np.where(df.columns.values == b)[0][0] for b in remove_correlations]
    i_nobinaries = list(set(i_nobinaries) - set(i_correlations))
    X = df.values[:, i_nobinaries]
    return X


