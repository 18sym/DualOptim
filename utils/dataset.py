from torchvision import transforms
from .cifar import CIFAR10, CIFAR100
from .datasets import isic2018
import os
import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset

def load_dataset(dataset):
    """
    Returns: dataset_train, dataset_test, num_classes
    """
    dataset_train = None
    dataset_test = None
    num_classes = 0

    if dataset == 'cifar10':
        trans_cifar10_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_cifar10_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = CIFAR10(
            root='./data/cifar',
            download=True,
            train=True,
            transform=trans_cifar10_train,
        )
        dataset_test = CIFAR10(
            root='./data/cifar',
            download=True,
            train=False,
            transform=trans_cifar10_val,
        )
        num_classes = 10

    elif dataset == 'cifar100':
        trans_cifar100_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])],
        )
        trans_cifar100_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])],
        )
        dataset_train = CIFAR100(
            root='./data/cifar100',
            download=True,
            train=True,
            transform=trans_cifar100_train,
        )
        dataset_test = CIFAR100(
            root='./data/cifar100',
            download=True,
            train=False,
            transform=trans_cifar100_val,
        )
        num_classes = 100
        
    elif dataset == 'isic2018':
        num_classes = 7
        train_root = "./data/ISIC_2018/ISIC2018_Training_Input"
        train_csv = "./data/ISIC_2018/train.csv"

        test_root = "./data/ISIC_2018/ISIC2018_Test_Input"
        test_csv = "./data/ISIC_2018/test.csv"

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        dataset_train = isic2018(root='./data/ISIC_2018/', transform=train_transform, mode='train')
        dataset_test = isic2018(root='./data/ISIC_2018/', transform=val_transform, mode='test')
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)

    else:
        raise NotImplementedError('Error: unrecognized dataset')

    return dataset_train, dataset_test, num_classes


