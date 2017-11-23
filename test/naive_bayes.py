# Import libraries
import numpy as np    # Numeric and matrix computation
import pandas as pd   # Optional: good package for manipulating data
import sklearn as sk  # Package with learning algorithms implemented


def test_profe():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    df = pd.read_csv(url,header =None)
    print(type(df))
    print(df.head())

    # No preprocessing needed. Numerical and scaled data
    # Separate data from labels

    y=df[34].values
    # print(y)
    X=df.values[:,0:34]

    from sklearn.model_selection import cross_val_score
    #from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import VotingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV

    cv=50

    clf1 = GaussianNB()

    params = {'n_neighbors':list(range(1,30,2)), 'weights':('distance','uniform')}
    knc = KNeighborsClassifier()
    clf = GridSearchCV(knc, param_grid=params,cv=cv,n_jobs=-1)  # If cv is integer, by default is Stratifyed
    clf.fit(X, y)
    print("Best Params fo Knn=",clf.best_params_, "Accuracy=", clf.best_score_)
    parval=clf.best_params_
    clf2 = KNeighborsClassifier(n_neighbors=parval['n_neighbors'],weights=parval['weights'])

    clf3 = DecisionTreeClassifier(criterion='entropy')


    for clf, label in zip([clf1, clf2, clf3], ['Naive Bayes','Knn (3)', 'Dec. Tree', ]):
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        print("Accuracy: %0.3f [%s]" % (scores.mean(), label))


def test_naive():
    path = '../dataset_diabetes/diabetic_data_output.csv'
    df = pd.read_csv(path)
    #print (df.corr())
    def get_redundant_pairs(df):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i + 1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def get_top_abs_correlations(df, n=5):
        au_corr = df.corr().abs().unstack()
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]

    print("Top Absolute Correlations")
    print(get_top_abs_correlations(df, 10))
    #print (df.values[:,[1,6,7,8,9]])
    # insulin
    # admission_type_id, discharge_disposition_id
    # admission_source_id
    # time_in_hospital
    # print(df["insulin"].values)
    #
    # print(df.values[:,0:34])
    # print(type(df.values))
    # print(type(df.columns.values.tolist().index("insulin")))
    # print(df.columns.values.tolist().index("insulin"))
    # print(type(np.where(df.columns.values == "insulin")))
    # print(np.where(df.columns.values == "insulin"))

if __name__ == "__main__":
    test_naive()