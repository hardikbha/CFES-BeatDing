import os
import sys
import time
import json
import random
import argparse
from collections import Counter
from itertools import cycle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from geomloss import SamplesLoss
from itertools import cycle
import util.aux_funcs as af
from src.modules import *
from src import logger
from src.metrics import *
from src.sam import SAM
from src.dataprocess import *
from src.transform import resnet_default_data_transforms as data_transforms
from detectors import DETECTOR
from torch.nn.modules.batchnorm import _BatchNorm
sys.path.append('.')
# from src.fairness_metrics import *
from torch.utils.data import DataLoader, Dataset
import random
from itertools import zip_longest
from fvcore.nn import FlopCountAnalysis


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


parser = argparse.ArgumentParser(description='Original model Training')
parser.add_argument('--seed', default=-1, type=int, help='seed for initializing training.')
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                    help='initial learning rate', dest='lr')
parser.add_argument('--result_dir', default='sink_results', type=str)
parser.add_argument('--dataset_dir', default='../data/dfdc/', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--workers', default=10, type=int)
parser.add_argument('--model_name', default='XceptionDetector', type=str,
                    help='Res50Detector, XceptionDetector')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

    dir_path = args.result_dir
    os.makedirs(dir_path, exist_ok=True)
    logger.log(f'===> random seed: {args.seed}')
    logger.configure(dir=dir_path, log_suffix='train')
    with open(os.path.join(dir_path, 'args_train.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_class = DETECTOR[args.model_name]
    model = model_class().to(device)

    # 计算 FLOPs
    input_shape = (1, 3, 256, 256)
    input_tensor = torch.randn(*input_shape).to(device)
    input_data = {
        'image': input_tensor,
        'label': torch.randint(0, 2, (input_shape[0],)).to(device),
        'intersec_label': torch.randint(0, 8, (input_shape[0],)).to(device),
    }
    flops = FlopCountAnalysis(model, input_data)
    total_flops = flops.total()
    print(f"模型总FLOPs: {total_flops / 1e9:.2f} GFlops")

    # 数据集与 DataLoader
    real_dataset = fair_train_dateset(
        dataset_dir=args.dataset_dir,
        owntransforms=data_transforms['train'],
        csv_file=os.path.join(args.dataset_dir, 'realtrain.csv'),
    )
    fake_dataset = fair_train_dateset(
        dataset_dir=args.dataset_dir,
        owntransforms=data_transforms['train'],
        csv_file=os.path.join(args.dataset_dir, 'faketrain.csv'),
    )
    combined_dataset = CombinedDataset(real_dataset, fake_dataset)

    combined_dataloader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )


    logger.log(
        f'# of real training samples: {len(real_dataset)}, '
        f'# of fake training samples: {len(fake_dataset)}, '
    )

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-3)

    for epoch in range(args.num_epochs):
        cur_lr = af.get_lr(optimizer)
        print(f'Epoch: {epoch}/{args.num_epochs}')
        print(f'Cur lr: {cur_lr}')

        model.train()
        avg_loss = 0.0
        total_batches = 0
        num_batches = len(combined_dataloader)

        for batch_idx, (real_data, fake_data) in enumerate(combined_dataloader):
            real_img, real_label, real_attr = real_data
            fake_img, fake_label, fake_attr = fake_data

            combined_img = torch.cat([real_img, fake_img], dim=0)
            combined_label = torch.cat([real_label, fake_label], dim=0)
            combined_attr = torch.cat([real_attr, fake_attr], dim=0)

            combined_data_dict = {
                'image': combined_img.to(device),
                'label': combined_label.to(device),
                'intersec_label': combined_attr.to(device),
            }

            preds = model(combined_data_dict)
            total_loss = model.get_losses(combined_data_dict, preds)['overall']

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            avg_loss += total_loss.item()
            total_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"Batch [{batch_idx + 1}/{num_batches}]  Total Loss: {total_loss.item():.4f}")

        epoch_avg_loss = avg_loss / max(total_batches, 1)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Avg Loss: {epoch_avg_loss:.4f}")

        # 保存模型
        torch.save(model.state_dict(), os.path.join(dir_path, f"{epoch}.pth"))
