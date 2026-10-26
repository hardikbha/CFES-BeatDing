import torch
import time
# import data
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import aux_funcs as af
from torch.optim import lr_scheduler
from util.custom_loss import *
# from data import *
from tqdm import tqdm
from model import *
from util.bce_acc import *
from sklearn import metrics
import torch.nn.functional as F
# from fairness_metric import *
from torchvision import models
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision
import clip
from torch.utils.data import DataLoader
import itertools
import os
from detectors import DETECTOR
from dataset.pair_dataset import pairDataset
from transform import resnet_default_data_transforms as data_transforms
from src.dataprocess import *
from geomloss import SamplesLoss
import torch.nn.utils.prune as prune
from src import logger

# vanilla scores (us to calculate FATE score)
# Fitz
ACC_B = 0.5513
FC_B = 0.1636

# ISIC
ACC_B = 0.7862
FC_B = 0.0209

def compute_fate(acc_m, acc_b, fc_m, fc_b, lamb=1.0):
    fate = ( (acc_m - acc_b) / acc_b ) - ( lamb * (fc_m - fc_b) / fc_b )
    return fate


class ChannelPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "channel"

    def __init__(self, channels):
        self.channels = channels

    def compute_mask(self, weight):
        mask = torch.ones_like(weight)
        for channel in self.channels:
            mask[:, channel * 7 * 7: (channel + 1) * 7 * 7] = 0 
            # mask[self.channels] = 0
        # print("mask:", mask)
        return mask


class OutChannelPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "channel"

    def __init__(self, channels):
        self.channels = channels

    def compute_mask(self, weight):
        mask = torch.ones_like(weight)
        for channel in self.channels:
            mask[channel, :, :, :] = 0
        # print("mask:", mask)
        return mask



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_cnn')
    parser.add_argument('--dataset_dir', default='../data/ff++', type=str)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--num_epochs', type=int, default=50,help='')
    parser.add_argument('--lr', type=float, default=0.001,help='')   
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sinkhorn_blur', default=5e-4, type=float)
    parser.add_argument('--lambda_fairloss', default=0.005, type=float)
    parser.add_argument('--result_dir', default='sink_results/', type=str)
    parser.add_argument('--batch_size', type=int, default=32,help='')
    parser.add_argument('--model_name', type=str, default='XceptionDetector',help='Res50Detector, XceptionDetector')
    parser.add_argument("--checkpoints", type=str,default="sink_results/XceptionDetector_ori/49.pth")
    parser.add_argument('--training_title', type=str, default='xception_prune_2%_only_head',help='')
    parser.add_argument('--channels_file', type=str, default=None,
                         help='DINGREPRO_LIN: path to a text file with one channel index per line, '
                              'from SFD.py min-Fk output. Falls back to Ding original hardcoded list if omitted.')
    args = parser.parse_args()
    training_title = args.training_title
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))
    device = af.get_pytorch_device()

    tensor_board_path = 'runs/{}/train_models{}'.format(
        training_title, af.get_random_seed())

    af.create_path(tensor_board_path)
    af.create_path('outputs/{}'.format(training_title))
    af.set_logger(
        'outputs/{}/train_models{}'.format(training_title, af.get_random_seed()))

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_class = DETECTOR[args.model_name]
    model = model_class()

    if cuda:
        model.cuda()
    ckpt = torch.load(args.checkpoints, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt, strict=True)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.backbone.last_linear.parameters():   
        p.requires_grad = True
    

    from fvcore.nn import FlopCountAnalysis


    input_shape = (1, 3, 256, 256)  
    input_tensor = torch.randn(*input_shape).to(device)  

    input_data = {
        'image': input_tensor,
        'label': torch.randint(0, 2, (input_shape[0],)).to(device),  
        'intersec_label': torch.randint(0, 8, (input_shape[0],)).to(device)  
    }

    flops = FlopCountAnalysis(model, input_data)  

    total_flops = flops.total()
    print(f"Total FLOPs: {total_flops / 1e9:.2f} GFlops")
    real_dataset = fair_train_dateset(dataset_dir=args.dataset_dir, owntransforms=data_transforms['train'],
                                      csv_file=os.path.join(args.dataset_dir, 'realtrain.csv'))
    fake_dataset = fair_train_dateset(dataset_dir=args.dataset_dir, owntransforms=data_transforms['train'],
                                      csv_file=os.path.join(args.dataset_dir, 'faketrain.csv'))
    combined_dataset = CombinedDataset(real_dataset, fake_dataset)
    combined_dataloader = DataLoader(combined_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=False)
    num_fake_batches = (len(fake_dataset) + args.batch_size - 1) // args.batch_size

    if args.channels_file:
        with open(args.channels_file) as f:
            channels = [int(x) for x in f.read().split()]
        print("DINGREPRO_LIN: loaded", len(channels), "channels to decouple from", args.channels_file)
    else:
        channels = [497, 275, 493, 95, 396, 13, 10, 391, 65, 431]  # Ding original hardcoded (Res50, their data)
    layer = find_last_adjust_channel_conv_layer(model)
    prune.custom_from_mask(layer, name="weight", mask=OutChannelPruning(channels).compute_mask(layer.weight))


    real_group_datasets, group0_sizes = create_group_dataloaders(dataset_dir=args.dataset_dir,
                                                                 csv_file=os.path.join(args.dataset_dir,'realtrain.csv'),
                                                                 data_transforms=data_transforms,
                                                                 batch_size=args.batch_size, num_workers=args.workers)
    print("The real data volume of each group:", group0_sizes)

    fake_group_datasets, group1_sizes = create_group_dataloaders(dataset_dir=args.dataset_dir,
                                                                 csv_file=os.path.join(args.dataset_dir,'faketrain.csv'),
                                                                 data_transforms=data_transforms,
                                                                 batch_size=args.batch_size, num_workers=args.workers)
    print("The fake data volume of each group:", group1_sizes)

    merged_group_datasets = []
    for real_dataset, fake_dataset in zip(real_group_datasets, fake_group_datasets):
        merged_dataset = MergedDataset(real_dataset, fake_dataset)
        merged_group_datasets.append(merged_dataset)

    dataloaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True
        )
        for dataset in merged_group_datasets
    ]
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-03)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.9)

    fairLoss = SamplesLoss(loss="sinkhorn", p=2, blur=args.sinkhorn_blur)

    # DINGREPRO_LIN fast-loader (Amendment 4a, D2 option b; the ONLY change vs GDA.py, see FASTLOADER_NOTE.md):
    # the released step called next(iter(selected_dataloader)), i.e. a NEW DataLoader iterator (8 fresh workers,
    # 16 prefetched batches) for every one 32+32 group batch. Here each group keeps one persistent iterator that is
    # re-created (re-shuffled) when its pass is exhausted. The short last batch of a pass is dropped, so every group
    # batch stays 32 real + 32 fake as in the released step. Two global-RNG draws are made when an existing iterator is
    # reused, the same as the replaced iter() call (worker base seed + RandomSampler seed), so the global torch RNG
    # stream, and hence the combined-batch order of every later epoch, is unchanged.
    group_iters = [None] * len(dataloaders)

    def next_group_batch(g):
        it = group_iters[g]
        if it is not None:
            batch = next(it, None)
            if batch is not None and len(batch[0][1]) == min(args.batch_size, len(dataloaders[g].dataset)):
                torch.empty((), dtype=torch.int64).random_()
                torch.empty((), dtype=torch.int64).random_()
                return batch
        group_iters[g] = None
        group_iters[g] = iter(dataloaders[g])
        return next(group_iters[g])

    for epoch in range(args.num_epochs):
        cur_lr = af.get_lr(optimizer)
        start_time = time.time()
        print(f'Epoch: {epoch}/{args.num_epochs}')
        print(f'Cur lr: {cur_lr}')
        model.train()
        avg_loss = 0.0
        total_batches = 0

        for batch_idx, (real_data, fake_data) in enumerate(combined_dataloader):
            real_img, real_label, real_attr = real_data
            fake_img, fake_label, fake_attr = fake_data

            combined_img = torch.cat([real_img, fake_img], dim=0)
            combined_label = torch.cat([real_label, fake_label], dim=0)
            combined_attr = torch.cat([real_attr, fake_attr], dim=0)
            combined_data_dict = {
                'image': combined_img.to(device),
                'label': combined_label.to(device),
                'intersec_label': combined_attr.to(device)
            }

            combined_preds = model(combined_data_dict)
            main_loss = model.get_losses(combined_data_dict, combined_preds)['overall']

            group_imgs, group_labels, group_attrs = [], [], []
            group_idx = random.randint(0, 7)  # DINGREPRO_LIN: corrected - true 8 groups (Others included), matches Ding
            selected_dataloader = dataloaders[group_idx]
            real_batch, fake_batch = next_group_batch(group_idx)  # DINGREPRO_LIN fast-loader; was next(iter(selected_dataloader))
            real_img_dist, real_label_dist, real_attr_dist = real_batch
            fake_img_dist, fake_label_dist, fake_attr_dist = fake_batch
            group_imgs = torch.cat([real_img_dist, fake_img_dist], dim=0)
            group_labels = torch.cat([real_label_dist, fake_label_dist], dim=0)
            group_attrs = torch.cat([real_attr_dist, fake_attr_dist], dim=0)
            group_data_dict = {
                'image': group_imgs.to(device),
                'label': group_labels.to(device),
                'intersec_label': group_attrs.to(device)
            }
            with torch.no_grad():
                group_preds = model(group_data_dict)
                fair_loss = fairLoss(combined_preds['cls'], group_preds['cls'])
                total_loss = main_loss + args.lambda_fairloss * fair_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            avg_loss += total_loss.item()
            total_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"Batch [{batch_idx + 1}/{num_fake_batches}]")
                print(
                    f"Total Loss: {total_loss.item():.4f},main_loss Loss: {main_loss.item():.4f},fair_loss Loss: {fair_loss.item():.4f}")

        epoch_avg_loss = avg_loss / total_batches
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], Avg Loss: {epoch_avg_loss:.4f}")

        if (epoch + 1) % 1 == 0:
            savepath = os.path.join(
                args.result_dir,
                f"{args.model_name}_prune2%_only_head",
                f"lr_{args.lr:.1e}_lambda_{args.lambda_fairloss:.1e}"
            )
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            temp_model = savepath + "/" + str(epoch) + '.pth'
            torch.save(model, temp_model)


