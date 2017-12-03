import numpy as np                     # Llibreria matemÃ tica
import matplotlib.pyplot as plt        # Per mostrar plots
import sklearn                         # Llibreia de DM
import sklearn.datasets as ds            # Per carregar mÃ©s facilment el dataset digits
import sklearn.model_selection as cv    # Pel Cross-validation
import sklearn.neighbors as nb           # Per fer servir el knn
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import cross_val_score  
from sklearn.model_selection import cross_val_predict  
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
#from statsmodels.stats.proportion import proportion_confint

def Method1(X, y):

    print("++++++++Executing method 1++++++++")


    cv_scores = cross_val_score(nb.KNeighborsClassifier(), X=X, y=y, cv=10, scoring='accuracy')  
    print(cv_scores)
    print(np.mean(cv_scores))
    print(np.std(cv_scores))

    # cv_scores is a list with 10 accuracies (one for each validation)
    print("**Printing a list with 10 accuracies (one for each validation)")
    
    print("\n")

    print("**Printing the mean of the 10 validations (and standard deviation of them)")
    print(np.mean(cv_scores))
    print(np.std(cv_scores))
    print("\n")



def Method2(X, y, y_test, y_pred):

    print("++++++++Executing method 2++++++++")
    # Build confussion matrix of all 10 cross-validations
    predicted = cross_val_predict(nb.KNeighborsClassifier(), X=X, y=y,  cv=10)  


    print(sklearn.metrics.confusion_matrix(y, predicted))
    print(sklearn.metrics.accuracy_score(y, predicted))
    print("\n")

    # In[44]:
    confmat = sklearn.metrics.confusion_matrix(y, predicted)

    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center',fontsize=7)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    #plt.tight_layout()
    plt.savefig('KNN-Plots/ConMatrix.png', dpi=600)
    plt.show()


    # In[12]:
    print(metrics.classification_report(y_test, y_pred))



def authomatically_best_parameters(X_train, y_train):

    lr = []
    for ki in range(1,200,2):
        cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=ki), X=X_train, y=y_train, cv=10)
        lr.append(np.mean(cv_scores))
    plt.plot(range(1,200,2),lr,'b',label='No weighting')

    lr = []
    for ki in range(1,200,2):
        cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=ki,weights='distance'), X=X_train, y=y_train, cv=10)
        lr.append(np.mean(cv_scores))

    plt.plot(range(1,200,2),lr,'r',label='Weighting')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.grid()
    #plt.tight_layout()
    plt.show()
    plt.savefig('KNN-Plots/BestParams.png', dpi=600)


def MacTestClassifier(X, y):
    # Classifier 1 (3 Neighbours) successes
    y_pred = cross_val_predict(nb.KNeighborsClassifier(n_neighbors=21), X=X, y=y,  cv=10) 
    res1=np.zeros(y.shape)
    res1[y_pred==y]=1

    # Classifier 2 (7 Neighbours) 2 successes
    y_pred = cross_val_predict(nb.KNeighborsClassifier(n_neighbors=9), X=X, y=y,  cv=10) 
    res2=np.zeros(y.shape)
    res2[y_pred==y]=1

    # Build contingency matrix
    n00 = np.sum([res1[res2==1]==1])
    n11 = np.sum([res1[res2==0]==0])
    n10 = np.sum([res1[res2==1]==0])
    n01 = np.sum([res1[res2==0]==1])

    # Chi -square test
    print("Have the classifiers significant different accuracy?:",(np.abs(n01-n10)-1)**2/(n01+n10)>3.84)

    

def executeKNN(X, y):
    # Simple cross-validation: split data into training and test sets (test 30% of data)
    (X_train, X_test,  y_train, y_test) = cv.train_test_split(X, y, test_size=.3, random_state=1)

    knc = nb.KNeighborsClassifier()
    knc.fit(X_train, y_train)

    print("**Printing accuracy")
    print(knc.score(X_test, y_test))
    print("\n")

    y_pred = knc.predict(X_test)
    print("**Printing more information with confussion matrix")
    print(sklearn.metrics.confusion_matrix(y_test, y_pred))
    print("\n")


    print("**Printing Recall, Precision and F-Measure")
    print(metrics.classification_report(y_test, y_pred))

    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=75,weights='distance'), X=X_train, y=y_train, cv=10)
    print(sklearn.metrics.confusion_matrix(y, cv_scores))
    print(sklearn.metrics.accuracy_score(y, cv_scores))
    print(np.mean(cv_scores))

    Method1(X, y)
    Method2(X, y, y_test, y_pred)
    authomatically_best_parameters(X_train, y_train)
    #GridSearch(X_train, y_train, X_test, y_test)
    MacTestClassifier(X, y)

