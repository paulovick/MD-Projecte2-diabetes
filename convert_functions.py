
import re

def convert_gender(value):
    if value:
        if value == "Male":
            return 0
        else:
            return 1
    return None

def convert_age(value):
    regex_pattern = '\d+'
    result = re.findall(regex_pattern, value)[0]
    return result

def convert_admission_type(value):
    if value == 6 | value == 8:
        return None
    return value

def convert_change(value):
    if value:
        if value == "Ch":
            return True
        else:
            return False
    return None