# coding: utf-8
# # K-NN in python: search for the best k
# ## 1- Load the required modeules
# In[1]:
import numpy as np                     # Llibreria matemÃ tica
import matplotlib.pyplot as plt        # Per mostrar plots
import sklearn                         # Llibreia de DM
import sklearn.datasets as ds            # Per carregar mÃ©s facilment el dataset digits
import sklearn.model_selection as cv    # Pel Cross-validation
import sklearn.neighbors as nb           # Per fer servir el knn


# ## 2- Load the data

# In[2]:

def executeKNN(ds):
    # Load digits dataset from scikit
    # Separate data from labels
    X = np.array(ds.drop('readmitted',1))
    y = np.array(ds['readmitted'])
    # Print range of values and dimensions of data
    # Data and labels are numpy array, so we can use associated methods
    #print((X.min(), X.max()))
    # Images are 8x8 pixels.
    print("**Printing shape")
    print(X.shape)
    print("\n")

    '''
    # In[3]:
    # Just for demostration purposes, let's see some images.
    nrows, ncols = 2, 5
    plt.figure(figsize=(6,3));
    plt.gray()
    for i in range(ncols * nrows):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.matshow(digits.images[i,...])
        plt.xticks([]); plt.yticks([]);
        plt.title(digits.target[i]);
    plt.show()
    '''


    # ## 3- Simple cross-validation

    # In[4]:
    # Let's do a simple cross-validation: split data into training and test sets (test 30% of data)
    (X_train, X_test,  y_train, y_test) = cv.train_test_split(X, y, test_size=.3, random_state=1)

    # Create a kNN classifier object
    knc = nb.KNeighborsClassifier()

    # Train the classifier
    knc.fit(X_train, y_train)

    # Obtain accuracy score of learned classifier on test data
    print("**Printing accuracy score of learned classifier on test data")
    print(knc.score(X_test, y_test))
    print("\n")



    # In[5]:
    # More information with confussion matrix
    from sklearn.metrics import confusion_matrix

    y_pred = knc.predict(X_test)
    print("**Printing more information with confussion matrix")
    print(sklearn.metrics.confusion_matrix(y_test, y_pred))
    print("\n")

    # In[6]:
    # Obtain Recall, Precision and F-Measure for each class
    from sklearn import metrics

    print("**Printing Recall, Precision and F-Measure")
    print(metrics.classification_report(y_test, y_pred))
    print("\n")

    '''
    # ### Let's build a one by hand to see prediction
    # In[7]:
    one = np.zeros((8, 8))
    one[1:-1, 4] = 16  # The image values are in [0, 16].
    one[2, 3] = 16

    # Draw the artifical image we just created
    plt.figure(figsize=(2,2));
    plt.imshow(one, interpolation='none');
    plt.grid(False);
    plt.xticks(); plt.yticks();
    plt.title("One");
    plt.show()
    '''

    '''
    # In[8]:
    # Let's see prediction for the new image
    print(knc.predict(one.reshape(1, 64)))
    '''

    # ## 4- Let's do a 10-fold cross-validation

    # In[9]:
    # Method 1
    print("++++++++Executing method 1++++++++")
    from sklearn.model_selection import cross_val_score  
    from sklearn.model_selection import cross_val_predict  
    from sklearn.metrics import accuracy_score

    cv_scores = cross_val_score(nb.KNeighborsClassifier(),  
                                X=X,  
                                y=y,  
                                cv=10, scoring='accuracy')  

    # cv_scores is a list with 10 accuracies (one for each validation)
    print("**Printing a list with 10 accuracies (one for each validation)")
    print(cv_scores)
    print("\n")


    # In[10]:
    # Let's get the mean of the 10 validations (and standard deviation of them)
    print("**Printing the mean of the 10 validations (and standard deviation of them)")
    print(np.mean(cv_scores))
    print(np.std(cv_scores))
    print("\n")


    # In[11]:

    # Method 2
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

    plt.tight_layout()
    plt.savefig('KNN-Plots/ConMatrix.png', dpi=600)
    #plt.show()


    # In[12]:
    print(metrics.classification_report(y_test, y_pred))


    '''
    # ### [Optional] Let's see how ammount of training data influences accuracy 
    # In[13]:
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores =                learning_curve(estimator=nb.KNeighborsClassifier(n_neighbors=3),
                                   X=X,
                                   y=y,
                                   train_sizes=np.linspace(0.05, 1.0, 10),
                                   cv=10,
                                   n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid(True)
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.03])
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=600)
    plt.show()
    '''


    # ## 5- Finding parameters for k-NN

    # In[14]:


    # See parameters in
    # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    # Results with different parameters: k
    print("**Printing Cross-Validation scores depending on n_neighbors and weights")
    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=1), X=X_train, y=y_train,  cv=10)  
    print("Accuracy 1 neighbour:", np.mean(cv_scores))

    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=3), X=X_train, y=y_train,  cv=10)  
    print("Accuracy 3 neighbours:",  np.mean(cv_scores))

    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=5), X=X_train, y=y_train,  cv=10)  
    print("Accuracy 5 neighbours:",  np.mean(cv_scores))

    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=7), X=X_train, y=y_train,  cv=10)  
    print("Accuracy 7 neighbours:",  np.mean(cv_scores))

    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=9), X=X_train, y=y_train,  cv=10)  
    print("Accuracy 9 neighbours:",  np.mean(cv_scores))


    # In[15]:


    # Results with different parameters: k and distance weighting
    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=1,weights='distance'), X=X_train, y=y_train,  cv=10)  
    print("Accuracy 1 neighbour: and distance weighting:", np.mean(cv_scores))

    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=3,weights='distance'), X=X_train, y=y_train,  cv=10)  
    print("Accuracy 3 neighbour: and distance weighting:", np.mean(cv_scores))

    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=5,weights='distance'), X=X_train, y=y_train,  cv=10)  
    print("Accuracy 5 neighbour: and distance weighting:", np.mean(cv_scores))

    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=7,weights='distance'), X=X_train, y=y_train,  cv=10)  
    print("Accuracy 7 neighbour: and distance weighting:", np.mean(cv_scores))

    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=9,weights='distance'), X=X_train, y=y_train,  cv=10)  
    print("Accuracy 9 neighbour: and distance weighting:", np.mean(cv_scores))

    
    # ### Authomatically find best parameters:

    # In[16]:
    
    lr = []
    for ki in range(1,30,2):
        cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=ki), X=X_train, y=y_train, cv=10)
        lr.append(np.mean(cv_scores))
    plt.plot(range(1,30,2),lr,'b',label='No weighting')

    lr = []
    for ki in range(1,30,2):
        cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=ki,weights='distance'), X=X_train, y=y_train, cv=10)
        lr.append(np.mean(cv_scores))
    plt.plot(range(1,30,2),lr,'r',label='Weighting')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()

    plt.savefig('KNN-Plots/BestParams.png', dpi=600)

    plt.show()


    # ### Do the same using Grid Search method in python

    # In[17]:


    from sklearn.model_selection import GridSearchCV
    params = {'n_neighbors':list(range(1,30,2)), 'weights':('distance','uniform')}
    knc = nb.KNeighborsClassifier()
    clf = GridSearchCV(knc, param_grid=params,cv=10,n_jobs=-1)  # If cv is integer, by default is Stratifyed 
    clf.fit(X_train, y_train)
    print("Best Params=",clf.best_params_, "Accuracy=", clf.best_score_)


    # Apply models with best parameters found trained with all training data to the test set

    # In[18]:


    parval=clf.best_params_
    knc = nb.KNeighborsClassifier(n_neighbors=parval['n_neighbors'],weights=parval['weights'])
    knc.fit(X_train, y_train)
    pred=knc.predict(X_test)
    print(sklearn.metrics.confusion_matrix(y_test, pred))
    print(sklearn.metrics.accuracy_score(y_test, pred))


    # In[19]:


    # interval confidence
    from statsmodels.stats.proportion import proportion_confint

    epsilon = sklearn.metrics.accuracy_score(y_test, pred)
    print("Can approximate by Normal Distribution?: ",X_test.shape[0]*epsilon*(1-epsilon)>5)
    print("Interval 95% confidence:", "{0:.3f}".format(epsilon), "+/-", "{0:.3f}".format(1.96*np.sqrt(epsilon*(1-epsilon)/X_test.shape[0])))
    # or equivalent 
    proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='normal')


    # In[20]:


    #Using Binomial distribution

    proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test')


    # ### Mcnemar's Test implementation

    # In[21]:


    # Build two classifiers

    # Classifier 1 (3 Neighbours) successes
    y_pred = cross_val_predict(nb.KNeighborsClassifier(n_neighbors=3), X=X, y=y,  cv=10) 
    res1=np.zeros(y.shape)
    res1[y_pred==y]=1

    # Classifier 2 (7 Neighbours) 2 successes
    y_pred = cross_val_predict(nb.KNeighborsClassifier(n_neighbors=7), X=X, y=y,  cv=10) 
    res2=np.zeros(y.shape)
    res2[y_pred==y]=1

    # Build contingency matrix
    n00 = np.sum([res1[res2==1]==1])
    n11 = np.sum([res1[res2==0]==0])
    n10 = np.sum([res1[res2==1]==0])
    n01 = np.sum([res1[res2==0]==1])

    # Chi -square test
    print("Have the classifiers significant different accuracy?:",(np.abs(n01-n10)-1)**2/(n01+n10)>3.84)


    # ### Take a look to the errors in test set

    # In[22]:

'''
    testerrors=[i for i,k in enumerate(pred) if k!=y_test[i]]
    plt.gray()
    plt.ion
    for i in testerrors:
        plt.matshow(X_test[i].reshape(8,8))
        plt.xticks([]); plt.yticks([]);
        print("Guess:", pred[i],"Reality:",y_test[i])

        plt.show()

    # # Exercises:
    # 
    # ### Do a 10-fold Cross-Validation using Naive Bayes

    # In[23]:


    from sklearn.naive_bayes import GaussianNB  # For numerical featuresm assuming normal distribution
    from sklearn.naive_bayes import MultinomialNB  # For features with counting numbers (f.i. hown many times word appears in doc)
    from sklearn.naive_bayes import BernoulliNB  # For binari features (f.i. word appears or not in document)

    # No parameters to tune

    clf = GaussianNB()
    pred = clf.fit(X_train, y_train).predict(X_test)
    print(sklearn.metrics.confusion_matrix(y_test, pred))
    print()
    print("Accuracy:", sklearn.metrics.accuracy_score(y_test, pred))
    print()
    print(metrics.classification_report(y_test, pred))
    epsilon = sklearn.metrics.accuracy_score(y_test, pred)
    proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test')


    # ### Reproduce in Rapidminer

    # In[24]:


    # Export data to Rapidminer

    import pandas as pd 
    df = pd.DataFrame(np.c_[ digits.data, digits.target])
    df.to_csv("digits2.csv",index=False)

    # Go to Rapidminer and load the data set. Reproduce grid Search there and report results on the test set


    # ### Play with noise

    # In[25]:


    # Lets' add noise to data: 64 new columns with random data
    nrcols=64
    col = np.random.randint(0,17,(X_train.data.shape[0],nrcols))
    col


    # In[26]:


    Xr=np.hstack((X_train,col))
    Xr


    # In[27]:


    col = np.random.randint(0,17,(X_test.data.shape[0],nrcols))
    Xr_test=np.hstack((X_test,col))


    # In[28]:


    lr = []
    for ki in range(1,30,2):
        knc = nb.KNeighborsClassifier(n_neighbors=ki)
        knc.fit(X_train, y_train)
        lr.append(knc.score(X_test, y_test))         
    plt.plot(range(1,30,2),lr,'b',label='No noise')

    lr = []
    for ki in range(1,30,2):
        knc = nb.KNeighborsClassifier(n_neighbors=ki)
        knc.fit(Xr, y_train)
        lr.append(knc.score(Xr_test, y_test))  
    plt.plot(range(1,30,2),lr,'r',label='With noise')

    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()

    plt.show()


    # In[29]:


    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2, mutual_info_classif

    fs = SelectKBest(mutual_info_classif, k=64).fit(Xr, y_train) #chi2
    X_new = fs.transform(Xr)
    Xtr_new = fs.transform(Xr_test)


    # In[30]:


    lr = []
    for ki in range(1,30,2):
        knc = nb.KNeighborsClassifier(n_neighbors=ki)
        knc.fit(Xr, y_train)
        lr.append(knc.score(Xr_test, y_test))    
    plt.plot(range(1,30,2),lr,'r',label='With noise')

    lr = []
    for ki in range(1,30,2):
        knc = nb.KNeighborsClassifier(n_neighbors=ki)
        knc.fit(X_train, y_train)
        lr.append(knc.score(X_test, y_test))   
    plt.plot(range(1,30,2),lr,'b',label='No noise')

    lr = []
    for ki in range(1,30,2):
        knc = nb.KNeighborsClassifier(n_neighbors=ki)
        knc.fit(X_new, y_train)
        lr.append(knc.score(Xtr_new, y_test))   
    plt.plot(range(1,30,2),lr,'g',label='Noise removed')

    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()

    plt.show()


    # ### Change the scorer function

    # In[34]:


    #Let's try to optimize parameters for precision of class "9"

    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.metrics import make_scorer

    params = {'n_neighbors':list(range(1,30,2)), 'weights':('distance','uniform')}
    knc = nb.KNeighborsClassifier()
    clf = GridSearchCV(knc, param_grid=params,cv=10,n_jobs=-1,scoring="accuracy") 
    clf.fit(X_train, y_train)
    print("Best Params=",clf.best_params_, "Accuracy=", clf.best_score_)

    parval=clf.best_params_
    knc = nb.KNeighborsClassifier(n_neighbors=parval['n_neighbors'],weights=parval['weights'])
    knc.fit(X_train, y_train)
    pred=knc.predict(X_test)
    print(sklearn.metrics.confusion_matrix(y_test, pred))
    print(sklearn.metrics.accuracy_score(y_test, pred))
    print(metrics.classification_report(y_test, pred))

    print('Precision for "9": %.3f' % precision_score(y_true=y_test, y_pred=pred,average='macro',labels=[9]))


    # In[35]:


    # Precison of class 9 is low compared with others. 
    # Assume precision of "9" is critical. Let's change optimize parameters by defining precision for 9.

    scorer = make_scorer(precision_score,average='macro',labels=[9])
    #scorer = make_scorer(score_func=precision_score, pos_label=9, greater_is_better=True,average='micro')


    params = {'n_neighbors':list(range(1,30,2)), 'weights':('distance','uniform')}
    knc = nb.KNeighborsClassifier()
    clf = GridSearchCV(knc, param_grid=params,cv=10,n_jobs=-1,scoring=scorer) 
    clf.fit(X_train, y_train)
    print("Best Params=",clf.best_params_, "Precision=", clf.best_score_)

    parval=clf.best_params_
    knc = nb.KNeighborsClassifier(n_neighbors=parval['n_neighbors'],weights=parval['weights'])
    knc.fit(X_train, y_train)
    pred=knc.predict(X_test)
    print(sklearn.metrics.confusion_matrix(y_test, pred))
    print(sklearn.metrics.accuracy_score(y_test, pred))
    print(metrics.classification_report(y_test, pred))


    # In[36]:


    testerrors=[i for i,k in enumerate(pred) if k!=y_test[i]]
    plt.gray()
    plt.ion
    for i in testerrors:
        plt.matshow(X_test[i].reshape(8,8))
        plt.xticks([]); plt.yticks([]);
        print("Guess:", pred[i],"Reality:",y_test[i])
        plt.show()
'''
