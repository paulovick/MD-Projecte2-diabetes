
def find_rows_by_unique_values(dataset, column_name):
    def find_numbers(medical_specialty):
        num = 0
        for item in enumerate(dataset[column_name]):
            if item[1] == medical_specialty:
                num += 1
        print(str(medical_specialty) + ': ' + str(num))

    for speciality in dataset[column_name].unique():
        find_numbers(speciality)

def find_nones(dataset):
    for column in dataset.columns.values:
        num_nulls = 0
        for value in dataset[column]:
            if value == None:
                num_nulls += 1
        print(str(column) + ": " + str(num_nulls))
