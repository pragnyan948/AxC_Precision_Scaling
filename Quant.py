import numpy as np
import csv
import string

def bit_levels(value, row, train):
    if value.endswith('-bit'):
        row.append(value[0])
        if train:
            for _ in range(2):
                row.append('8') #weight/act, dx & dwprecision levels
    elif value.startswith('INT'):
        row.append(value[3]) #weight/act precision levels
        if train:
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

def Quant_Stat(value, row, train):
    prec = list(map(int, row[-3:]))
    #import pdb;pdb.set_trace()
    def max_quantization_error(precision_bits):
        max_error = 1 / (2 ** (precision_bits - 1))
        return max_error
    
    for i in range(2):
        row.append(max_quantization_error(prec[0]))

    if train:
        for i in range(2):
            row.append(max_quantization_error(prec[i+1]))
    #import pdb;pdb.set_trace()
    return row


def Quant_method(value, row, train):
    split=value.split('+')
    if train:
        number_methods =4
    else:
        number_methods =2
    for i in range(number_methods):
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