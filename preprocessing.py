
import pandas as pd
from convert_functions import *
import matplotlib.pyplot as plt
from pandas.tools import plotting
from scipy.optimize import curve_fit

def read_and_filter_dataset(use_preprocessed=False,preprocess=True,nrows=5000,save_csv=False):
    path = 'dataset_diabetes/diabetic_data.csv'
    dest_path = 'dataset_diabetes/diabetic_data_output.csv'

    if use_preprocessed:
        dataset = pd.read_csv(dest_path, na_values=['?','None','nan'])
        return dataset

    dataset = pd.read_csv(path, na_values=['?','None','nan'], nrows=nrows)

    if preprocess:
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
            'diag_1': convert_diag,
            'diag_2': convert_diag,
            'diag_3': convert_diag,
            'number_diagnoses': None,
            'change': convert_change,
            'diabetesMed': convert_diabetesMed,
            'readmitted': convert_readmitted
        }

        dataset = dataset.drop('encounter_id',1)
        dataset = dataset.drop('patient_nbr',1)
        dataset = dataset.drop('max_glu_serum',1)
        dataset = dataset.drop('A1Cresult',1)
        dataset = dataset.drop('weight',1)
        dataset = dataset.drop('payer_code',1)

        for column in dataset:
            if dataset[column].nunique() == 1:
                dataset = dataset.drop(column, 1)
            else:
                if column not in column_functions.keys():
                    dataset[column] = dataset[column].apply(convert_generic)
                elif column_functions[column] != None:
                    dataset[column] = dataset[column].apply(column_functions[column])
                else:
                    dataset[column] = dataset[column].apply(convert_base)

        # Erase nulls

        dataset = dataset.fillna(round(dataset.mean()))
        #print(dataset.isnull().any())

    # Save

    if save_csv:
        dataset.to_csv(dest_path, index=False)
    
    return dataset

def plot_statistics(dataset):
    plt.close('all')

    # Core data
    fig = plt.figure()

    genderAxis = plt.subplot2grid((2,2),(0,0))
    genderAxis.set_xlabel('Gender')
    ageAxis = plt.subplot2grid((2,2),(1,0))
    ageAxis.set_xlabel('Age Intervals')
    raceAxis = plt.subplot2grid((2,2),(0,1),rowspan=2)
    raceAxis.set_xlabel('Race')

    dataset.race.value_counts().plot(kind='bar',ax=raceAxis)
    dataset.gender.value_counts().plot(kind='bar',ax=genderAxis)
    dataset.age.value_counts().plot(kind='bar',ax=ageAxis)

    plt.tight_layout()
    plt.show()

    # Generics 1
    fig = plt.figure()

    metforminAxis = plt.subplot2grid((4,3),(0,0))
    metforminAxis.set_xlabel('Metformin',fontsize=9)
    repaglinideAxis = plt.subplot2grid((4,3),(0,1))
    repaglinideAxis.set_xlabel('Repaglinide',fontsize=9)
    nateglinideAxis = plt.subplot2grid((4,3),(0,2))
    nateglinideAxis.set_xlabel('Nateglinide',fontsize=9)
    chlorpropamideAxis = plt.subplot2grid((4,3),(1,0))
    chlorpropamideAxis.set_xlabel('Chlorpropamide',fontsize=9)
    glimepirideAxis = plt.subplot2grid((4,3),(1,1))
    glimepirideAxis.set_xlabel('Glimepiride',fontsize=9)
    acetohexamideAxis = plt.subplot2grid((4,3),(1,2))
    acetohexamideAxis.set_xlabel('Acetohexamide',fontsize=9)
    glipizideAxis = plt.subplot2grid((4,3),(2,0))
    glipizideAxis.set_xlabel('Glipizide',fontsize=9)
    glyburideAxis = plt.subplot2grid((4,3),(2,1))
    glyburideAxis.set_xlabel('Glyburide',fontsize=9)
    tolbutamideAxis = plt.subplot2grid((4,3),(2,2))
    tolbutamideAxis.set_xlabel('Tolbutamide',fontsize=9)
    pioglitazoneAxis = plt.subplot2grid((4,3),(3,0))
    pioglitazoneAxis.set_xlabel('Pioglitazone',fontsize=9)
    rosiglitazoneAxis = plt.subplot2grid((4,3),(3,1))
    rosiglitazoneAxis.set_xlabel('Rosiglitazone',fontsize=9)
    acarboseAxis = plt.subplot2grid((4,3),(3,2))
    acarboseAxis.set_xlabel('Acarbose',fontsize=9)

    dataset.metformin.value_counts().plot(kind='bar',ax=metforminAxis)
    dataset.repaglinide.value_counts().plot(kind='bar',ax=repaglinideAxis)
    dataset.nateglinide.value_counts().plot(kind='bar',ax=nateglinideAxis)
    dataset.chlorpropamide.value_counts().plot(kind='bar',ax=chlorpropamideAxis)
    dataset.glimepiride.value_counts().plot(kind='bar',ax=glimepirideAxis)
    dataset.acetohexamide.value_counts().plot(kind='bar',ax=acetohexamideAxis)
    dataset.glipizide.value_counts().plot(kind='bar',ax=glipizideAxis)
    dataset.glyburide.value_counts().plot(kind='bar',ax=glyburideAxis)
    dataset.tolbutamide.value_counts().plot(kind='bar',ax=tolbutamideAxis)
    dataset.pioglitazone.value_counts().plot(kind='bar',ax=pioglitazoneAxis)
    dataset.rosiglitazone.value_counts().plot(kind='bar',ax=rosiglitazoneAxis)
    dataset.acarbose.value_counts().plot(kind='bar',ax=acarboseAxis)

    plt.tight_layout()
    plt.show()

    # Generics 2
    fig = plt.figure()

    miglitolAxis = plt.subplot2grid((4,3),(0,0))
    miglitolAxis.set_xlabel('Miglitol',fontsize=9)
    troglitazoneAxis = plt.subplot2grid((4,3),(0,1))
    troglitazoneAxis.set_xlabel('Troglitazone',fontsize=9)
    tolazamideAxis = plt.subplot2grid((4,3),(0,2))
    tolazamideAxis.set_xlabel('Tolazamide',fontsize=9)
    examideAxis = plt.subplot2grid((4,3),(1,0))
    examideAxis.set_xlabel('Examide',fontsize=9)
    citogliptonAxis = plt.subplot2grid((4,3),(1,1))
    citogliptonAxis.set_xlabel('Citoglipton',fontsize=9)
    insulinAxis = plt.subplot2grid((4,3),(1,2))
    insulinAxis.set_xlabel('Insulin',fontsize=9)
    glyburideMetforminAxis = plt.subplot2grid((4,3),(2,0))
    glyburideMetforminAxis.set_xlabel('Glyburide Metformin',fontsize=9)
    glipizideMetforminAxis = plt.subplot2grid((4,3),(2,1))
    glipizideMetforminAxis.set_xlabel('Glipizide Metformin',fontsize=9)
    glimepiridePioglitazoneAxis = plt.subplot2grid((4,3),(2,2))
    glimepiridePioglitazoneAxis.set_xlabel('Glimepiride Pioglitazone',fontsize=9)
    metforminRosiglitazoneAxis = plt.subplot2grid((4,3),(3,0))
    metforminRosiglitazoneAxis.set_xlabel('Metformin Rosiglitazone',fontsize=9)
    metforminPioglitazoneAxis = plt.subplot2grid((4,3),(3,1))
    metforminPioglitazoneAxis.set_xlabel('Metformin Pioglitazone',fontsize=9)

    dataset.miglitol.value_counts().plot(kind='bar',ax=miglitolAxis)
    dataset.troglitazone.value_counts().plot(kind='bar',ax=troglitazoneAxis)
    dataset.tolazamide.value_counts().plot(kind='bar',ax=tolazamideAxis)
    dataset.examide.value_counts().plot(kind='bar',ax=examideAxis)
    dataset.citoglipton.value_counts().plot(kind='bar',ax=citogliptonAxis)
    dataset.insulin.value_counts().plot(kind='bar',ax=insulinAxis)
    dataset['glyburide-metformin'].value_counts().plot(kind='bar',ax=glyburideMetforminAxis)
    dataset['glipizide-metformin'].value_counts().plot(kind='bar',ax=glipizideMetforminAxis)
    dataset['glimepiride-pioglitazone'].value_counts().plot(kind='bar',ax=glimepiridePioglitazoneAxis)
    dataset['metformin-rosiglitazone'].value_counts().plot(kind='bar',ax=metforminRosiglitazoneAxis)
    dataset['metformin-pioglitazone'].value_counts().plot(kind='bar',ax=metforminPioglitazoneAxis)

    plt.tight_layout()
    plt.show()

    # Results
    fig = plt.figure()

    changeAxis = plt.subplot2grid((1,2),(0,0))
    changeAxis.set_xlabel('Change')
    diabetesAxis = plt.subplot2grid((1,2),(0,1))
    diabetesAxis.set_xlabel('Diabetes')

    dataset.change.value_counts().plot(kind='bar',ax=changeAxis)
    dataset.diabetesMed.value_counts().plot(kind='bar',ax=diabetesAxis)

    plt.tight_layout()
    plt.show()