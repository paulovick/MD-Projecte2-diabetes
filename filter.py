import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from convert_functions import *
from utils import *
from naive_bayes import naive_bayes

path = 'dataset_diabetes/diabetic_data.csv'
dest_path = 'dataset_diabetes/diabetic_data_output.csv'

column_functions = {
    'encounter_id': None,
    'patient_nbr': None,
    'race': convert_race,
    'gender': convert_gender,
    'age': convert_age,
    'admission_type_id': convert_admission_type,
    'discharge_disposition_id': convert_discharge_disposition,
    'admission_source_id': convert_admission_source,
    'time_in_hospital': None,
    'medical_specialty': convert_medical_specialty,
    'num_lab_procedures': None,
    'num_procedures': None,
    'num_medications': None,
    'number_outpatient': None,
    'number_emergency': None,
    'number_inpatient': None,
    'diag_1': None,
    'diag_2': None,
    'diag_3': None,
    'number_diagnoses': None,
    'max_glu_serum': None,
    'A1Cresult': None,
    'change': convert_change,
    'diabetesMed': convert_diabetesMed,
    'readmitted': convert_readmitted
}

dataset = pd.read_csv(path, na_values=['?','None','nan'])#, nrows=1000)

# find_rows_by_unique_values(dataset, 'medical_specialty')
# find_nones(dataset)

dataset = dataset.drop('weight',1)
dataset = dataset.drop('payer_code',1)

# for column in dataset:
#     if column not in column_functions.keys():
#         dataset[column] = dataset[column].apply(convert_generic)
#     elif column_functions[column] != None:
#         dataset[column] = dataset[column].apply(column_functions[column])
#     else:
#         dataset[column] = dataset[column].apply(convert_base)

naive_bayes(dataset)
# print(dataset.shape)
# X = np.array(dataset.drop('readmitted',1))
# y = np.array(dataset['readmitted'])

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