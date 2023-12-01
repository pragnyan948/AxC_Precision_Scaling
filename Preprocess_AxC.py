import csv
import numpy as np

def import_data(file_path):
    data_list = []
    
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        for row in csv_reader:
            data_list.append(row)

    return data_list

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

def preprocess_data(list_of_dicts):
    out=[]
    all_values = extract(list_of_dicts)
    #print("All key-value pairs:")
    #print('[wg prec, dx prec, dw prec, act_type, conv_layers, single_prec, act method, wg method, dx method, dw method, train, ISO accuracy]')
    out=[["wgprec", "dxprec", "dwprec", "acttype", "convlayers", "singleprec", "actmethod", "wgmethod", "dxmethod", "dwmethod", "train", "ISOaccuracy"]]
    for dictionary_items in all_values:
        row=[]
        for key, value in dictionary_items:
            if key.startswith('Dataset#'):
                Index=value
                #print(f'{key}: {value}')
            elif key.startswith('Data_'):
                row=bit_levels(value, row) #row would be appended by [weight/act prec, dx prec, dw prec]
                row=Quant_Stat(value, row)
                #if Index == '93':
                    #import pdb;pdb.set_trace()
                #print(f'{key}: {value}')
            elif key.startswith('Datas'):
                row=datset_stat(value,row)  
            elif key.startswith('Algo'):
                row=layer_info(value,row) #row would be appended by [number_conv_layers,number_act_layers, act_type]
            elif key.startswith('PE'):
                row.append(value)
            elif key.startswith('Quant'):
                row=Quant_method(value,row)
            elif key.startswith('Opti'):
                row=optim_info(value, row) #row would be appended by [train,res_neuron, res_wg]
            elif key.startswith('Iso'):
                if value.startswith('Y'):
                    row.append('1')
                else:
                    row.append('0')
        out.append(row)
    #import pdb;pdb.set_trace()
    print(np.array(out[0:2]))
    print(len(out))
    export_data(out)
    #import pdb;pdb.set_trace()
    return out



def bit_levels(value, row):
    if value.endswith('-bit'):
        for _ in range(3):
            row.append(value[0]) #weight/act, dx & dwprecision levels
    elif value.startswith('INT'):
        row.append(value[3]) #weight/act precision levels
        split=value.split('+')
        #import pdb;pdb.set_trace()
        #split=value.split(maxsplit=4,'+')
        if split[2].startswith('FP'):
            row.append(split[2][2]) # dx precision levels
            row.append(split[3][-1]) # dw precision levels
        else:
            for _ in range(2):
                row.append(split[2][-1]) # dx & dwprecision levels
    else:
        row.append('0')
    #row would be appended by [weight/act orec, dx prec, dw prec]
    return row

def layer_info(value,row):
    if value.startswith('Res'):
        row.append('RELU')
        row.append(value.split('-')[1])
        #import pdb;pdb.set_trace()
    else:
        row.append('NA')
        row.append('0')
    return row 


def Quant_Stat(value, row):
    return row

def datset_stat(value, row):
    return row

def Quant_method(value, row):
    split=value.split('+')
    for i in range(4):
        if split[i] == 'DOREFA':
            row.append('1')
        elif split[i] == 'FAQ':
            row.append('2')
        elif split[i] == 'LSQ_KD':
            row.append('3')
        elif split[i] == 'PACT':
            row.append('4')
        elif split[i] == 'Radix-4':
            row.append('5')
        elif split[i] == 'SWAB':
            row.append('6')
        elif split[i] == 'TPR':
            row.append('7')
        else:
            row.append('0')
    return row

def optim_info(value, row):
    split=value.split('+')
    if split[0] == 'QAT':
        row.append ('true')
    else:
        row.append ('false')
    return row


def export_data(data_list):
    # Specify the CSV file path
    csv_file_path = './Data/Data.csv'

    # Writing to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data_list)

    print(f'The list has been successfully written to {csv_file_path}.')