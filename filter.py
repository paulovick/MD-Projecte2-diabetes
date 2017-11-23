from naive_bayes import naive_bayes
from preprocessing import *

dataset = read_and_filter_dataset(preprocess=True,nrows=5000,save_csv=True)
#dataset = read_and_filter_dataset(use_preprocessed=True)

# plot_statistics(dataset)
# plot_null_statistics(dataset)

naive_bayes(dataset)

# from sklearn import preprocessing
# import matplotlib.pyplot as plt

# encoder = preprocessing.LabelEncoder()

# encoder.fit(y)
# # plt.scatter(X[:,0], y, c=encoder.transform(y), cmap=plt.cm.Paired)
# # plt.show()

# from sklearn import cross_validation
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
# from os import system

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
# # print(X_train.shape)
# # print(X_test.shape)

# from sklearn.tree import DecisionTreeClassifier

# clf = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
# clf.score(X_test, y_test)     
# y_pred = clf.predict(X_test)
# print "Classification Report:"
# print metrics.classification_report(y_test, y_pred)
# print "Confusion Matrix:"
# print metrics.confusion_matrix(y_test, y_pred)