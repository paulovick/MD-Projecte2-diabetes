import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from convert_functions import convert_age, convert_change

path = 'dataset_diabetes/diabetic_data.csv'
dest_path = 'dataset_diabetes/diabetic_data_output.csv'

# headers = ['encounter_id','patient_nbr','race','gender','age','weight','admission_type_id','discharge_disposition_id','admission_source_id','time_in_hospital','payer_code','medical_specialty','num_lab_procedures','num_procedures','num_medications','number_outpatient','number_emergency','number_inpatient','diag_1','diag_2','diag_3','number_diagnoses','max_glu_serum','A1Cresult','metformin','repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone','change','diabetesMed','readmitted']

dataset = pd.read_csv(path, na_values=['?','None'], nrows=10)

dataset['age'] = dataset['age'].apply(convert_age)
dataset['change'] = dataset['change'].apply(convert_change)

dataset.to_csv(dest_path)

# print(dataset.shape)
# X = np.array(dataset[['race','gender','age','weight','admission_type_id','discharge_disposition_id','admission_source_id','time_in_hospital']])
# y = np.array(dataset['diabetesMed'])

# from sklearn import preprocessing
# import matplotlib.pyplot as plt

# encoder = preprocessing.LabelEncoder()

# encoder.fit(y)
# plt.scatter(X[:,0], y, c=encoder.transform(y), cmap=plt.cm.Paired)
# plt.show()

# from sklearn import cross_validation
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
# from os import system

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
# print(X_train.shape)
# print(X_test.shape)

# from sklearn.tree import DecisionTreeClassifier

# clf = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
# clf.score(X_test, y_test)     
# y_pred = clf.predict(X_test)
# print "Classification Report:"
# print metrics.classification_report(y_test, y_pred)
# print "Confusion Matrix:"
# print metrics.confusion_matrix(y_test, y_pred)