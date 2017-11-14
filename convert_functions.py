
import re

def convert_base(value):
    return value

def convert_generic(value):
    value = convert_base(value)
    if value != None:
        if value == 'No':
            return 0
        elif value == 'Steady':
            return 1
        elif value == 'Up':
            return 2
        elif value == 'Down':
            return 3
    return None

def convert_race(value):
    value = convert_base(value)
    if value != None:
        if value == 'Caucasian':
            return 0
        elif value == 'AfricanAmerican':
            return 1
        elif value == 'Asian':
            return 2
        elif value == 'Hispanic':
            return 3
        elif value == 'Other':
            return 4
    return None

def convert_gender(value):
    value = convert_base(value)
    if value != None:
        if value == 'Male':
            return 0
        elif value == 'Female':
            return 1
    return None

def convert_age(value):
    value = convert_base(value)
    if value != None:
        regex_pattern = '\d+'
        result = re.findall(regex_pattern, value)[0]
        return result
    return None

def convert_admission_type(value):
    value = convert_base(value)
    if value in [5, 6, 8]:
        return None
    return value

def convert_discharge_disposition(value):
    value = convert_base(value)
    if value in [18, 25, 26]:
        return None
    return value

def convert_admission_source(value):
    value = convert_base(value)
    if value in [15, 17, 20, 21]:
        return None
    return value

def convert_medical_specialty(value):
    value = convert_base(value)
    if value != None:
        if value == 'Pediatrics-Endocrinology':
            return 0
        elif value == 'InternalMedicine':
            return 1
        elif value == 'Family/GeneralPractice':
            return 2
        elif value == 'Cardiology':
            return 3
        elif value == 'Surgery-General':
            return 4
        elif value == 'Orthopedics':
            return 5
        elif value == 'Gastroenterology':
            return 6
        elif value == 'Surgery-Cardiovascular/Thoracic':
            return 7
        elif value == 'Nephrology':
            return 8
        elif value == 'Orthopedics-Reconstructive':
            return 9
        elif value == 'Psychiatry':
            return 10
        elif value == 'Emergency/Trauma':
            return 11
        elif value == 'Pulmonology':
            return 12
        elif value == 'Surgery-Neuro':
            return 13
        elif value == 'Obsterics&Gynecology-GynecologicOnco':
            return 14
        elif value == 'ObstetricsandGynecology':
            return 15
        elif value == 'Pediatrics':
            return 16
        elif value == 'Hematology/Oncology':
            return 17
        elif value == 'Otolaryngology':
            return 18
        elif value == 'Surgery-Colon&Rectal':
            return 19
        elif value == 'Pediatrics-CriticalCare':
            return 20
        elif value == 'Endocrinology:':
            return 21
        elif value == 'Urology':
            return 22
        elif value == 'Psychiatry-Child/Adolescent':
            return 23
        elif value == 'Pediatrics-Pulmonology':
            return 24
        elif value == 'Neurology':
            return 25
        elif value == 'Anesthesiology-Pediatric':
            return 26
        elif value == 'Radiology':
            return 27
        elif value == 'Pediatrics-Hematology-Oncology':
            return 28
        elif value == 'Psychology':
            return 29
        elif value == 'Podiatry':
            return 30
        elif value == 'Gynecology':
            return 31
        elif value == 'Oncology':
            return 32
        elif value == 'Pediatrics-Neurology':
            return 33
        elif value == 'Surgery-Plastic':
            return 34
        elif value == 'Surgery-Thoracic':
            return 35
        elif value == 'Surgery-PlasticwithinHeadandNeck':
            return 36
        elif value == 'Ophthalmology':
            return 37
        elif value == 'Surgery-Pediatric':
            return 38
        elif value == 'Pediatrics-EmergencyMedicine':
            return 39
        elif value == 'PhysicalMedicineandRehabilitation':
            return 40
        elif value == 'InfectiousDiseases':
            return 41
        elif value == 'Anesthesiology':
            return 42
        elif value == 'Rheumatology':
            return 43
        elif value == 'AllergyandImmunology':
            return 44
        elif value == 'Surgery-Maxillofacial':
            return 45
        elif value == 'Pediatrics-InfectiousDiseases':
            return 46
        elif value == 'Pediatrics-AllergyandImmunology':
            return 47
        elif value == 'Dentistry':
            return 48
        elif value == 'Surgeon':
            return 49
        elif value == 'Surgery-Vascular':
            return 50
        elif value == 'Osteopath':
            return 51
        elif value == 'Psychiatry-Addictive':
            return 52
        elif value == 'Surgery-Cardiovascular':
            return 53
        elif value == 'PhysicianNotFound':
            return 54
        elif value == 'Hematology':
            return 55
        elif value == 'Proctology':
            return 56
        elif value == 'Obstetrics':
            return 57
        elif value == 'SurgicalSpecialty':
            return 58
        elif value == 'Radiologist':
            return 59
        elif value == 'Pathology':
            return 60
        elif value == 'Dermatology':
            return 61
        elif value == 'SportsMedicine':
            return 62
        elif value == 'Speech':
            return 63
        elif value == 'Hospitalist':
            return 64
        elif value == 'OutreachServices':
            return 65
        elif value == 'Cardiology-Pediatric':
            return 66
        elif value == 'Perinatology':
            return 67
        elif value == 'Neurophysiology':
            return 68
        elif value == 'Endocrinology-Metabolism':
            return 69
        elif value == 'DCPTEAM':
            return 70
        elif value == 'Resident':
            return 71
    return None

def convert_change(value):
    value = convert_base(value)
    if value != None:
        if value == 'Ch':
            return True
        elif value == 'No':
            return False
    return None

def convert_diabetesMed(value):
    value = convert_base(value)
    if value != None:
        if value == 'Yes':
            return True
        elif value == 'No':
            return False
    return None

def convert_readmitted(value):
    value = convert_base(value)
    if value != None:
        if value == 'NO':
            return 0
        elif value == '<30':
            return 1
        elif value == '>30':
            return 2
    return None