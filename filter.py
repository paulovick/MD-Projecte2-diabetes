# from naive_bayes import naive_bayes
# from svm import svm
# from KNN import executeKNN
from decisionTree import decisionTree
from preprocessing import *
from KNN import *
from adaboost import *


def splitting(ds, nrows):
    y_name = "readmitted"  # volem predir la columna readmitted, aixi que sera la nostre y
    X = ds.drop(['readmitted'], axis=1)
    y = ds[y_name]
    (X1, Xresta, y1, yresta) = cv.train_test_split(X, y, train_size=nrows, random_state=1, stratify=y)
    return X1, y1


dataset = read_and_filter_dataset(preprocess=True, save_csv=True)
# dataset = read_and_filter_dataset(use_preprocessed=True)
X, y = splitting(dataset, nrows=5000)

# plot_statistics(dataset)
# plot_null_statistics(dataset)

# naive_bayes(dataset)
# svm(dataset)
if __name__ == "__main__":
    # sense la linia de dalt, executeKNN quedara en loop infinit
    # executeKNN(dataset)
    # decisionTree(dataset)
    adaboost(dataset)
