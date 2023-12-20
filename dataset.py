import numpy as np
import csv
import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def datset_stat(value, row):
    #your_dataset_loader = DataLoader(torch.randn((100, 3, 32, 32)), batch_size=32, shuffle=True)
    your_dataset_loader=load_dataset(value)
    #if value.startswith('NLP'):
        #import pdb;pdb.set_trace()

    results = analyze_dataset(your_dataset_loader)
    for i in range(4):
        row.append(results[i])

    print(f"Max Value: {results[0]}")
    print(f"Mean of Absolute Values: {results[1]}")
    print(f"Variance of Absolute Values: {results[2]}")
    print(f"Number of Positive Elements: {results[3]}")

    return row
    

def analyze_dataset(dataset_loader):
    max_value = 0 
    sum_absolute = 0.0
    sum_squared_absolute = 0.0
    num_positive_elements = 0
    num_samples =0

    for batch in dataset_loader:
        data = batch[0]  # Assuming the data is in the first element of each batch
        max_value = max(max_value, torch.max(torch.abs(data)).item())
        sum_absolute += torch.sum(torch.abs(data)).item()
        sum_squared_absolute += torch.sum(torch.abs(data) ** 2).item()
        num_positive_elements += torch.sum(data > 0).item()
        num_samples += torch.numel(data)
    
    mean_absolute = sum_absolute / num_samples
    variance_absolute = (sum_squared_absolute / num_samples) - (mean_absolute ** 2)
    num_positive_elements =num_positive_elements*1e-9

    return max_value, mean_absolute, variance_absolute, num_positive_elements

def load_dataset(dataset_name):
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    elif dataset_name.lower() == 'svhn':
        transform = transforms.Compose([transforms.ToTensor()])
        data = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    elif dataset_name.lower() == 'imagenet':
        imagenet_path = './Data/ImageNet-Datasets-Downloader-master/imagenet_images'
        # Define the transform to apply to each image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        data = ImageFolder(root=imagenet_path, transform=transform)

    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized.")

    data_loader = DataLoader(data, batch_size=64, shuffle=True)
    return data_loader