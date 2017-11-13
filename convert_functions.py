def convert_age(value):
    regex_pattern = '\d+'
    result = re.findall(regex_pattern, value)[0]
    return result

def convert_change(value):
    return value