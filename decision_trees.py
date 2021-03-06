import pandas as pd
import numpy as np
import sklearn.model_selection as cv
import sklearn
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot
from IPython.display import Image
from statsmodels.stats.proportion import proportion_confint
from sklearn.model_selection import cross_val_score  
import matplotlib.pyplot as plt

def decision_trees(X, y):

    (X_train, X_test, y_train, y_test) = cv.train_test_split(X, y, test_size=.3, random_state=0)

    clf = tree.DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = sklearn.metrics.accuracy_score(y_test, pred)
    confmat = sklearn.metrics.confusion_matrix(y_test, pred)

    print("Confusion matrix:")
    print(confmat)
    print()
    print("Accuracy on test set: ", score)
    print()
    print(sklearn.metrics.classification_report(y_test, pred))
    print("Confidence interval: ", proportion_confint(count=score*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test'))

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

    # means = []
    # for i in range(2,50):
    #     dtc = tree.DecisionTreeClassifier(criterion="entropy")
    #     dtc.fit(X, y)
        
    #     cv_scores = cross_val_score(dtc, X=X, y=y, cv=i, scoring='accuracy') 
    #     #print (cv_scores)
    #     #print(np.mean(cv_scores))
    #     #print(np.std(cv_scores))

    #     means.append(np.mean(cv_scores))
    
    # plt.plot(means)
    # plt.ylabel('Accuracy')
    # plt.xlabel('K')
    # plt.show()

    # dot_data = StringIO()
    # tree.export_graphviz(clf, out_file=dot_data,
    #                    filled=True, rounded=True,
    #                    feature_names=list(X.columns.values),
    #                    special_characters=True)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # # Image(graph.create_png())
    # tree.export_graphviz(dtc, out_file='plots/treepic.dot', feature_names=X.columns)
