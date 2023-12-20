import numpy as np
import csv
import torch
import torch.nn as nn
import torchvision.models as models
import tensorflow as tf
from tensorflow.keras import applications

def analyze_model_tf(model_name):
    import pdb;pdb.set_trace()
    try:
        model = getattr(applications, model_name)(weights=None)
    except AttributeError:
        print(f"Model '{model_name}' not found in tensorflow.keras.applications.")
        return

    def count_layers(model):
        conv_layer_count = 0
        activation_layer_count = 0
        activation_types = set()

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layer_count += 1
            elif isinstance(layer, tf.keras.layers.Activation):
                activation_layer_count += 1
                activation_types.add(layer.__class__.__name__)

        return conv_layer_count, activation_layer_count, activation_types
    counts = count_layers(model)

    return counts

def get_model_info_torch(model_name):
    # Load the pre-trained model
    #import pdb;pdb.set_trace()
    try:
        model = getattr(models, model_name)(pretrained=True)
    except AttributeError:
        print(f"Model '{model_name}' not found. Make sure it is a valid torchvision model.")
        return

    conv_layer_count = 0
    activation_layer_count = 0
    activation_type = set()

    def count_layers(module):
        nonlocal conv_layer_count, activation_layer_count
        for layer in module.children():
            if isinstance(layer, nn.Conv2d):
                conv_layer_count += 1
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU) or isinstance(layer, nn.ELU) :
                activation_layer_count += 1
                activation_type.add(layer.__class__.__name__)
            elif list(layer.children()):
                # Recursive call for nested modules
                count_layers(layer)
    count_layers(model)
    print(f"Number of Convolution Layers: {conv_layer_count}")
    print(f"Number of Activation Layers: {activation_layer_count}")
    print(f"Activation Types: {', '.join(activation_type)}")
    #import pdb;pdb.set_trace()
    return conv_layer_count, activation_layer_count,sorted(activation_type)



def layer_info(value,row):
    #import pdb;pdb.set_trace()
    if value.startswith('torch'):
        conv_layer_count, activation_layer_count,activation_type=get_model_info_torch(value.split('-')[1])
        row.append(conv_layer_count)
        row.append(activation_layer_count)
        #if(activation_layer_count==0):
            #row.append('NA')
        #else:
            #row.append(activation_type[0])
    elif value.startswith('tf'):
        conv_layer_count, activation_layer_count,activation_type=analyze_model_tf(value.split('-')[1])
        row.append(conv_layer_count)
        row.append(activation_layer_count)
        #row.append(activation_type)
    elif value.startswith('res'):
        #import pdb;pdb.set_trace()
        row.append(str(int(value[-2:])-2))
        row.append(str(int(value[-2:])-1))
        #row.append('RELU')
        #import pdb;pdb.set_trace()
    elif value.startswith('SVHN'):
        row.append('7')
        row.append('8')
        #row.append('RELU')
    else:
        row.append('0')
        row.append('0')
        row.append('NA')
    """
    row.append(conv_layer_count)
    row.append(activation_layer_count)
    row.append(activation_type)
    
    if value.startswith('Res'):
        row.append('RELU')
        row.append(value.split('-')[1])
        #import pdb;pdb.set_trace()
    else:
        row.append('NA')
        row.append('0')
    """
    return row 
