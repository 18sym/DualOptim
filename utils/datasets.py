from PIL import Image
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from .sampling import sample_dirichlet
import torch.utils
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.utils.data as data
from torchvision.datasets.folder import default_loader

def get_dataset(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'cifar10':
        data_path = '../data/cifar10'
        args.num_classes = 10
        # args.model = 'resnet18'
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = datasets.CIFAR10(
            data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR10(
            data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)
    elif args.dataset == 'cifar100':
        data_path = '../data/cifar100'
        args.num_classes = 100
        args.model = 'resnet34'
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])],
        )
        dataset_train = datasets.CIFAR100(
            data_path, train=True, download=True, transform=trans_train)
        dataset_test = datasets.CIFAR100(
            data_path, train=False, download=True, transform=trans_val)
        n_train = len(dataset_train)
        y_train = np.array(dataset_train.targets)


    else:
        exit('Error: unrecognized dataset')

    # For Real-world experiments on Clothing1M
    # if args.dataset == 'clothing1m':
    #     pass

    if(args.partition == 'dirichlet'):
        print(
            "@@@@@@@@@@@@@         Non-IID data partitioning via Dirichlet distribution")
        dict_users = sample_dirichlet(
            y_train, args.num_classes, args.non_iid_prob_class, args.num_users, args.seed, args.alpha_dirichlet)

    return dataset_train, dataset_test, dict_users

class isic2018(torch.utils.data.Dataset):
    def __init__(self, root, transform, mode):
        self.root = root
        self.data = []
        self.targets = []
        self.transform = transform
        self.mode = mode

        if self.mode == 'train':
            csv_file = self.root + 'train.csv'
        elif self.mode == 'test':
            csv_file = self.root + 'test.csv'
        else:
            raise ValueError("mode should be 'train' or 'test'")

        annotations = pd.read_csv(csv_file)

        for idx, row in annotations.iterrows():
            img_name = row['image']
            img_path = f"{self.root}/ISIC2018_Test_Input/{img_name}.jpg" if self.mode == 'test' else f"{self.root}/ISIC2018_Training_Input/{img_name}.jpg"
            self.data.append(img_path)

            label_index = torch.argmax(torch.tensor(row[1:].values.astype('float'))).item()
            self.targets.append(label_index)

        self.train_labels = np.array(self.targets) if self.mode == 'train' else None
        self.test_labels = np.array(self.targets) if self.mode == 'test' else None

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        image = Image.open(img_path).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.data)