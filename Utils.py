import csv
import numpy as np

def import_data(file_path):
    data_list = []
    
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            data_list.append(row)

    return data_list

def export_data(data_list, train):
    # Specify the CSV file path
    if train:
        csv_file_path = './Data_AxC/Data_train.csv'
    else:
        csv_file_path = './Data_AxC/Data_inf.csv'

    # Writing to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_list)

    print(f'The list has been successfully written to {csv_file_path}.')

def extract(list_of_dicts):
    """
    Extracts all key-value pairs from a list of dictionaries.

    Parameters:
    - list_of_dicts (list): The input list of dictionaries.

    Returns:
    - A list of dictionaries where each dictionary contains key-value pairs.
    """
    all_values_list = []

    for dictionary in list_of_dicts:
        all_values_list.append(dictionary.items())

    return all_values_list