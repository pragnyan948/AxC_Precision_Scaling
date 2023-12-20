import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from model import layer_info
from Quant import bit_levels, Quant_method, Quant_Stat
from dataset import datset_stat
from Utils import import_data, export_data, extract

def preprocess_data(list_of_dicts, train):
    out=[]
    all_values = extract(list_of_dicts)
    #print("All key-value pairs:")
    #print('[wg prec, dx prec, dw prec, act_type, conv_layers, single_prec, act method, wg method, dx method, dw method, train, ISO accuracy]')
    if train:
        out=[["wgprec", "dxprec", "dwprec","wgmaxqerr","actmaxqerr","dxmaxqerr","dwmaxqerr","maxds","meands","vards","signds" ,"convlayers", "actlayers", "actmethod", "wgmethod", "dxmethod", "dwmethod", "ISOaccuracy"]]
    else:
        out=[["wgprec", "wgmaxqerr","actmaxqerr","maxds","meands","vards","signds" ,"convlayers", "actlayers", "actmethod", "wgmethod", "ISOaccuracy"]]
    for dictionary_items in all_values:
        row=[]
        for key, value in dictionary_items:
            if key.startswith('Dataset#'):
                Index=value
                print(f'{key}: {value}')
            elif key.startswith('Data_'):
                row=bit_levels(value, row, train) #row would be appended by [weight/act prec, dx prec, dw prec]
                row=Quant_Stat(value, row, train)
                #if Index == '93':
                    #import pdb;pdb.set_trace()
                #print(f'{key}: {value}')
            elif key.startswith('Datas'):
                row=datset_stat(value,row)  
            elif key.startswith('Algo'):
                print(f'{key}: {value}')
                row=layer_info(value,row) #row would be appended by [number_conv_layers,number_act_layers, act_type]
            #elif key.startswith('PE'):
                #row.append(value)
            elif key.startswith('Quant'):
                row=Quant_method(value,row, train)
            #elif key.startswith('Opti'):
                #row=optim_info(value, row) #row would be appended by [train]
            elif key.startswith('Iso'):
                if value.startswith('Y'):
                    row.append('1')
                else:
                    row.append('0')
        out.append(row)
    #import pdb;pdb.set_trace()
    print(np.array(out[0:2]))
    print(len(out))
    export_data(out, train)
    #import pdb;pdb.set_trace()
    return out


def optim_info(value, row):
    split=value.split('+')
    if split[0] == 'QAT':
        row.append ('true')
    else:
        row.append ('false')
    return row

def transform_X_Y(train):
    if train:
        dataset = pd.read_csv('./Data_AxC/Data_train.csv')
    else:
        dataset = pd.read_csv('./Data_AxC/Data_inf.csv')
    dataset.isna().sum()
    #dataset.info()
    X = dataset.drop('ISOaccuracy', axis=1)
    d = {'X': X}
    print(d)
    y = dataset['ISOaccuracy']
    d = {'Y': y}
    print(d)

    # Columns to be moved to the end
    categorical_features_X = []

    # Reorder columns by moving specified columns to the end
    #X = X[[col for col in X.columns if col not in categorical_features_X] + categorical_features_X]

    # Display the reordered DataFrame
    print(X)

    one_hot = OneHotEncoder()
    transformer = ColumnTransformer([("one_hot",
                                    one_hot,
                                    categorical_features_X)],
                                    remainder="passthrough")

    transformed_X = transformer.fit_transform(X)
    d = {'X': pd.DataFrame(transformed_X).head()}
    #print(pd.DataFrame(transformed_X).head())
    #import pdb;pdb.set_trace()

    X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size = 0.25, random_state = 2)

    return transformed_X, X_train, X_test, y_train, y_test 