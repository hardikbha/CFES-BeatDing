import os
import numpy as np
import random
from PIL import Image
import math
import copy
import pandas as pd
import re

import clip

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.models import *
import torch.nn.functional as F

from sklearn.metrics import *
# from fairlearn.metrics import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from torch.utils.data import DataLoader, Dataset

class CombinedDataset(Dataset):
    def __init__(self, real_dataset, fake_dataset):
        self.real_dataset = real_dataset
        self.fake_dataset = fake_dataset

    def __len__(self):
        return len(self.fake_dataset)

    def __getitem__(self, idx):
        fake_data = self.fake_dataset[idx]

        real_idx = random.randint(0, len(self.real_dataset) - 1)
        real_data = self.real_dataset[real_idx]
        return real_data, fake_data
class MergedDataset(Dataset):
    def __init__(self, real_dataset, fake_dataset):
        self.real_dataset = real_dataset
        self.fake_dataset = fake_dataset
        self.real_length = len(real_dataset)
        self.fake_length = len(fake_dataset)

    def __len__(self):
        return  self.fake_length

    def __getitem__(self, idx):
        real_idx = random.randint(0, self.real_length - 1)
        fake_idx = random.randint(0, self.fake_length - 1)

        real_sample = self.real_dataset[real_idx]
        fake_sample = self.fake_dataset[fake_idx]

        return real_sample, fake_sample

def find_last_adjust_channel_conv_layer(model):
    target_layer_name = None
    target_layer = None

    if hasattr(model, 'backbone') and hasattr(model.backbone, 'adjust_channel'):
        adjust_channel = model.backbone.adjust_channel
    elif hasattr(model, 'adjust_channel'):
        adjust_channel = model.adjust_channel

    for name, module in adjust_channel.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module
    return  target_layer

# DINGREPRO_LIN: corrected 2026-10-15 - raw Lin annotation (via EXACTMATCH_plan) DOES have Others; true 8 groups, matches Ding's own scheme.
def create_group_dataloaders(dataset_dir, csv_file, data_transforms, batch_size, num_workers, num_groups=8):
    group_datasets = []
    group_sizes = []

    for group_id in range(num_groups):
        group_dataset = fair_train_dateset(
            dataset_dir=dataset_dir,
            owntransforms=data_transforms['train'],
            csv_file=csv_file,
            target_group=group_id
        )
        group_dataloader = DataLoader(
            group_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        group_sizes.append(len(group_dataset))
        group_datasets.append(group_dataset)

    return group_datasets, group_sizes

class fair_dateset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='',owntransforms='',model='' ):
        self.transform = owntransforms
        if model == 'train':
            csv_file = os.path.join(dataset_dir, 'ff++', 'train.csv')
        elif model == 'test_ff++':
            csv_file = os.path.join(dataset_dir, 'ff++', 'test.csv')
        elif model == 'test_dfdc':
            csv_file = os.path.join(dataset_dir, 'dfdc', 'test.csv')
        elif model == 'test_celebdf':
            csv_file = os.path.join(dataset_dir, 'celebdf', 'test.csv')
        elif model == 'test_dfd':
            csv_file = os.path.join(dataset_dir, 'dfd', 'test.csv')
        else:
            raise ValueError("dataset_dir must correct!!")
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['img_path'])  # DINGREPRO_LIN: our csv img_path is already absolute, no lstrip
        try:
            img = Image.open(img_path).convert('RGB')
            img_trans = self.transform(img)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

        label = int(row['label'].item())
        intersec_label = int(row['intersec_label'].item())
        return img_trans, label,intersec_label

class fair_group(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='',owntransforms='', attribute='intersec_label', thegroup=0):
        self.transform = owntransforms
        csv_file = os.path.join(dataset_dir, 'ff++', 'train.csv')
        self.data = pd.read_csv(csv_file)
        tmp_files = []
        self.group_counts = {}
        for index, row in self.data.iterrows():
            group = int(row[attribute])
            if group == thegroup:
                tmp_files.append(row)
        self.files = tmp_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        row = self.files[idx]
        img_path = str(row['img_path'])  # DINGREPRO_LIN: our csv img_path is already absolute, no lstrip
        try:
            img = Image.open(img_path).convert('RGB')
            img_trans = self.transform(img)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

        label = int(row['label'])
        intersec_label = int(row['intersec_label'])

        return img_trans, label, intersec_label


class fair_train_dateset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir='', owntransforms='', csv_file='', target_group=None):
        self.transform = owntransforms
        self.data = pd.read_csv(csv_file)

        if target_group is not None:
            if 'realtrain.csv' in csv_file:
                new_filename = f"{target_group}_realtrain.csv"
            elif 'faketrain.csv' in csv_file:
                new_filename = f"{target_group}_faketrain.csv"
            new_csv_path = os.path.join(dataset_dir, new_filename)
            self.data = pd.read_csv(new_csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['img_path'])  # DINGREPRO_LIN: our csv img_path is already absolute, no lstrip
        try:
            img = Image.open(img_path).convert('RGB')
            img_trans = self.transform(img)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None, None, None

        label = int(row['label'])
        intersec_label = int(row['intersec_label'])
        return img_trans, label, intersec_label

class fair_dateset(torch.utils.data.Dataset):
    def __init__(self, csv_file='', owntransforms=''):
        self.transform = owntransforms
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row['img_path'])
        try:
            img = Image.open(img_path).convert('RGB')
            img_trans = self.transform(img)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None, None, None

        label = int(row['label'])
        intersec_label = int(row['intersec_label'])
        return img_trans, label, intersec_label