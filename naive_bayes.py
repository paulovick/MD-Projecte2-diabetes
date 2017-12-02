import numpy as np  # Llibreria matemÃ tica
import sklearn.model_selection as cv  # Pel Cross-validation
from sklearn.ensemble import VotingClassifier
# interval confidence
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB  # For binari features (f.i. word appears or not in document)
from sklearn.naive_bayes import GaussianNB  # For numerical featuresm assuming normal distribution
from sklearn.naive_bayes import \
    MultinomialNB  # For features with counting numbers (f.i. hown many times word appears in doc)


# Obtain Recall, Precision and F-Measure for each class
from statsmodels.stats.proportion import proportion_confint


def naive_bayes(X, y):
    X_total = X
    X_bernoulli = bernoulli(X)
    X_multinomial, i_multinomial = multinomial(X)
    X_gaussian = gaussian(X, i_multinomial)
    X_array = (("Bernoulli", X_bernoulli), ("Multinomial", X_multinomial), ("Gaussian", X_gaussian))

    estimators = []
    vc = 50
    for method, X in X_array:
        # Let's do a simple cross-validation: split data into training and test sets (test 30% of data)
        (X_train, X_test, y_train, y_test) = cv.train_test_split(X, y, test_size=.3, random_state=1, stratify=y)

        # No parameters to tune

        if method == "Bernoulli":
            clf = BernoulliNB()
        elif method == "Multinomial":
            clf = MultinomialNB()
        else:
            clf = GaussianNB()
        #clf.fit(X_train, y_train)
        scores = cross_val_score(clf, X, y, cv=vc, scoring='accuracy')
        print("%s Accuracy ( %s variables): %0.3f" % (method, len(X[0]), scores.mean()))

        scores = cross_val_score(clf, X_total, y, cv=vc, scoring='accuracy')
        print("%s Accuracy total ( %s variables): %0.3f" % (method, len(X[0]), scores.mean()))
        estimators.append((method, clf))

        pred = clf.fit(X_train, y_train).predict(X_test)
        unique, counts = np.unique(y_test, return_counts=True)
        print(dict(zip(unique, counts)))
        print(sklearn.metrics.confusion_matrix(y_test, pred, labels=[0, 1]))
        print()
        print("Accuracy:", sklearn.metrics.accuracy_score(y_test, pred))
        print()
        print(sklearn.metrics.classification_report(y_test, pred))
        epsilon = sklearn.metrics.accuracy_score(y_test, pred)
        print(proportion_confint(count=epsilon * X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test'))


    (X_train, X_test, y_train, y_test) = cv.train_test_split(X_total, y, test_size=.3, random_state=1, stratify=y)
    eclf = VotingClassifier(estimators=estimators, voting='hard')
    scores = cross_val_score(eclf, X_train, y_train, cv=vc, scoring='accuracy')
    print("Accuracy: %0.3f [%s]" % (scores.mean(), "Majority Voting"))
    pred = clf.fit(X_train, y_train).predict(X_test)
    unique, counts = np.unique(y_test, return_counts=True)
    print(dict(zip(unique, counts)))
    print(sklearn.metrics.confusion_matrix(y_test, pred, labels=[0,1]))
    print()
    print("Accuracy:", sklearn.metrics.accuracy_score(y_test, pred))
    print()
    print(sklearn.metrics.classification_report(y_test, pred))
    epsilon = sklearn.metrics.accuracy_score(y_test, pred)
    print(proportion_confint(count=epsilon * X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test'))
    confmat = sklearn.metrics.confusion_matrix(y_test, pred)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center', fontsize=7)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.tight_layout()
    plt.savefig('ConMatrix.png', dpi=600)
    plt.show()

def bernoulli(df):
    # Binaries
    nom_binaries = [col for col in df if df[col].nunique() == 2]
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


# print(sklearn.metrics.confusion_matrix(y_test, pred))
# print()
# print("%s acurracy:" % (method), sklearn.metrics.accuracy_score(y_test, pred))
# print()
# print(metrics.classification_report(y_test, pred))
# epsilon = sklearn.metrics.accuracy_score(y_test, pred)
# proportion_confint(count=epsilon * X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test')