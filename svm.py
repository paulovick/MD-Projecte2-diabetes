import numpy as np                     # Llibreria matemÃ tica
import matplotlib.pyplot as plt        # Per mostrar plots
import sklearn                         # Llibreia de DM
import sklearn.datasets as ds            # Per carregar mÃ©s facilment el dataset digits
import sklearn.model_selection as cv    # Pel Cross-validation
import sklearn.neighbors as nb           # Per fer servir el knn
import pandas as pd   # Optional: good package for manipulating data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

def svm(df):
    y_name = "readmitted"
    y = df[y_name].values
    i, = np.where(df.columns.values == y_name)  # index column of y value
    X = np.delete(df.values, i, 1).astype('float32')
    # normalize data (-1,1)
    (X_train, X_test, y_train, y_test) = cv.train_test_split(X, y, test_size=.3, stratify=y, random_state=1) #separem per a training i validation
    #print(df.describe())

    # scaler = StandardScaler().fit(X_train)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)

    # Apply the normalization trained in training data in both training and test sets
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # linear SVM
    # knc = LinearSVC()
    knc = SVC(kernel='linear')
    knc.fit(X_train, y_train)
    pred = knc.predict(X_test)
    print("Confusion matrix on test set:\n", sklearn.metrics.confusion_matrix(y_test, pred))
    print("\nAccuracy on test set: ", sklearn.metrics.accuracy_score(y_test, pred))

    knc = SVC(kernel='poly', degree=2)
    knc.fit(X_train, y_train)
    pred = knc.predict(X_test)
    print("Confusion matrix on test set:\n", sklearn.metrics.confusion_matrix(y_test, pred))
    print("\nAccuracy on test set: ", sklearn.metrics.accuracy_score(y_test, pred))

    knc = SVC()
    knc.fit(X_train, y_train)
    pred = knc.predict(X_test)
    print("Confusion matrix on test set:\n", sklearn.metrics.confusion_matrix(y_test, pred))
    print("\nAccuracy on test set: ", sklearn.metrics.accuracy_score(y_test, pred))

    #findCLineal(X_train, X_test, y_train, y_test)

    #findCPoly(X_train, X_test, y_train, y_test)

    #findC(X_train, y_train)


def findCLineal(X_train, X_test, y_train, y_test):
    # List of C values to test. We usualy test diverse orders of magnitude
    # Cs = np.logspace(-3, 11, num=15, base=10.0)
    Cs = np.logspace(-3, 5, num=9, base=10.0)

    param_grid = {'C': Cs}
    # grid_search = GridSearchCV(LinearSVC(), param_grid, cv=10)
    grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, n_jobs=-1, verbose=1, cv=10)
    grid_search.fit(X_train, y_train)

    # Let's plot the 10-fold cross.validation accuracy deppending on C
    scores = grid_search.cv_results_['mean_test_score']
    plt.semilogx(Cs, scores)
    plt.show()

    # Let's apply the best C parameter found to the test set
    parval = grid_search.best_params_
    knc = LinearSVC(C=parval['C'])
    knc = SVC(C=parval['C'], kernel='linear')
    knc.fit(X_train, y_train)
    pred = knc.predict(X_test)
    print("Confusion matrix on test set:\n", sklearn.metrics.confusion_matrix(y_test, pred))
    print("\nAccuracy on test set: ", sklearn.metrics.accuracy_score(y_test, pred))
    print("\nBest value of parameter C found: ", parval)
    print("\nNumber of supports: ", np.sum(knc.n_support_), "(", np.sum(np.abs(knc.dual_coef_) == parval['C']),"of them have slacks)")
    print("Prop. of supports: ", np.sum(knc.n_support_) / X_train.shape[0])

    knc = SVC(kernel='poly', degree=2)
    knc.fit(X_train, y_train)
    pred = knc.predict(X_test)
    print("Confusion matrix on test set:\n", sklearn.metrics.confusion_matrix(y_test, pred))
    print("\nAccuracy on test set: ", sklearn.metrics.accuracy_score(y_test, pred))

def findCPoly(X_train, X_test, y_train, y_test):
    Cs = np.logspace(-2, 4, num=7, base=10.0)

    param_grid = {'C': Cs}
    grid_search = GridSearchCV(SVC(kernel='poly', degree=2), param_grid, cv=10)
    grid_search.fit(X_train, y_train)

    scores = grid_search.cv_results_['mean_test_score']

    plt.semilogx(Cs, scores)
    plt.show()

    parval = grid_search.best_params_
    knc = SVC(kernel='poly', degree=2, C=parval['C'])
    knc.fit(X_train, y_train)
    pred = knc.predict(X_test)
    print("Confusion matrix on test set:\n", sklearn.metrics.confusion_matrix(y_test, pred))
    print("\nAccuracy on test set: ", sklearn.metrics.accuracy_score(y_test, pred))
    print("\nBest combination of parameters found: ", parval)
    print("\nNumber of supports: ", np.sum(knc.n_support_), "(", np.sum(np.abs(knc.dual_coef_) == parval['C']),
          "of them have slacks)")
    print("Prop. of supports: ", np.sum(knc.n_support_) / X_train.shape[0])

def findC(X_train, y_train):
    # Values we will test for each parameter. When observin results, consider the limits of the
    # values tested and increase them if necessary
    gammas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    Cs = np.logspace(-1, 6, num=8, base=10.0)

    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(), param_grid, cv=10)
    grid_search.fit(X_train, y_train)
    parval = grid_search.best_params_

    # We'll show in a grid, the accuracy for each combination of parameters tester
    scores = grid_search.cv_results_['mean_test_score']
    scores = np.array(scores).reshape(len(param_grid['C']), len(param_grid['gamma']))

    plt.matshow(scores)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(param_grid['gamma'])), param_grid['gamma'], rotation='vertical')
    plt.yticks(np.arange(len(param_grid['C'])), param_grid['C'])
    plt.show()
    print("\nBest combination of parameters found: ", parval)