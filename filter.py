#from naive_bayes import naive_bayes
# from svm import svm
from KNN import executeKNN
# from decisionTree import decisionTree
from preprocessing import *
from KNN import *
from adaboost import *


def splitting(ds, nrows):
    y_name = "readmitted"  # value to predict
    y = ds[y_name]
    #Count y_value
    unique, counts = np.unique(y, return_counts=True)
    d_temp = dict(zip(unique, counts/len(y)*100))
    d = {k:round(float(v), 2) for k,v in d_temp.items()}
    print("% of value to predict ({}) BEFORE splitting dataset: {}".format(y_name, d))

    X = ds.drop(['readmitted',], axis=1)

    (X1, Xresta, y1, yresta) = cv.train_test_split(X, y, train_size=nrows, random_state=1, stratify=y)

    # Count y_value
    unique, counts = np.unique(y1, return_counts=True)
    d_temp = dict(zip(unique, counts / len(y1) * 100))
    d = {k: round(float(v), 2) for k, v in d_temp.items()}
    print("% of value to predict ({}) AFTER splitting dataset: {}".format(y_name, d))
    return X1, y1


#dataset = read_and_filter_dataset(preprocess=True, save_csv=True)
dataset = read_and_filter_dataset(use_preprocessed=True)
X, y = splitting(dataset, nrows=5000)

# plot_statistics(dataset)
# plot_null_statistics(dataset)


if __name__ == "__main__":
    # sense la linia de dalt, executeKNN quedara en loop infinit
    
    Xaux = X
    '''
    i = 0
    
    for column in X:
        print (column)
        X = Xaux
        X = X.drop(column, axis=1)
        executeKNN(X, y)
        ++i
    '''
    X = Xaux
    X = X.drop(['num_lab_procedures',  'age', 'number_diagnoses', 'time_in_hospital'], axis=1)
    print ("******")
    executeKNN(X, y)

        
        # naive_bayes(X, y)
        # decisionTree(dataset)
        # svm(dataset)
        # adaboost(dataset)
