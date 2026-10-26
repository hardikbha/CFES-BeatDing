import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from detectors import DETECTOR
from src.dataprocess import fair_dateset
from transform import resnet_default_data_transforms as data_transforms


class SoftNearestNeighborLoss(nn.Module):
    def __init__(self,
                 temperature=1,
                 cos_distance=True):
        super(SoftNearestNeighborLoss, self).__init__()

        self.temperature = temperature
        self.cos_distance = cos_distance

    def pairwise_cos_distance(self, A, B):
        query_embeddings = torch.nn.functional.normalize(A, dim=1)
        key_embeddings = torch.nn.functional.normalize(B, dim=1)
        distances = 1 - torch.matmul(query_embeddings, key_embeddings.T)
        return distances

    def forward(self, embeddings, labels):
        batch_size = embeddings.shape[0]
        eps = 1e-9

        pairwise_dist = self.pairwise_cos_distance(embeddings, embeddings)
        pairwise_dist = pairwise_dist / self.temperature
        negexpd = torch.exp(-pairwise_dist)

        # creating mask to sample same class neighboorhood
        batch_size = labels.size(0)
        pairs_y = labels.repeat(batch_size, 1)
        #         pairs_y = torch.broadcast_to(labels, (batch_size, batch_size))
        mask = pairs_y == torch.transpose(pairs_y, 0, 1)
        mask = mask.float()

        # creating mask to exclude diagonal elements
        ones = torch.ones([batch_size, batch_size], dtype=torch.float32).cuda()
        dmask = ones - torch.eye(batch_size, dtype=torch.float32).cuda()

        # all class neighborhood
        alcn = torch.sum(torch.multiply(negexpd, dmask), dim=1)
        # same class neighborhood
        sacn = torch.sum(torch.multiply(negexpd, mask), dim=1)

        # adding eps for numerical stability
        # in case of a class having a single occurance in batch
        # the quantity inside log would have been 0
        loss = -torch.log((sacn + eps) / alcn).mean()
        return loss


def extract_feature_from_512channel_conv(x, model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "adjust_channel"):
        target_layer = model.backbone.adjust_channel[0]
    elif hasattr(model, "adjust_channel"):
        target_layer = model.adjust_channel[0]
    else:
        raise AttributeError(
            "Model does not have 'adjust_channel' layer or it is not properly defined."
        )

    if hasattr(model, "backbone"):
        model = model.backbone
    elif hasattr(model, "resnet"):
        model = model.resnet

    layers = list(model.children())[:-2]
    feature_extractor = nn.Sequential(*layers)

    intermediate_features = feature_extractor(x)
    feature_map = target_layer(intermediate_features)  # (B, 512, 1, 1)
    return feature_map


def get_min_n(data, n):
    data_tmp = data.copy()
    data_tmp.sort()
    return data_tmp[n]


parser = argparse.ArgumentParser(description="")
parser.add_argument("--dataset_dir", default="/data1/ywh/Dataset/ff++", type=str)
parser.add_argument("--workers", default=4, type=int)
parser.add_argument("--batch_size", default=64, type=int)
# DINGREPRO_LIN: parameterized (official script hardcoded Res50Detector + a fixed checkpoint path + fixed n=10)
parser.add_argument("--model_name", default="XceptionDetector", type=str)
parser.add_argument("--checkpoint", default="sink_results/Res50Detector_ori/49.pth", type=str)
parser.add_argument("--prc", default=2.0, type=float, help="percent of 512 adjust_channel channels to decouple (Ding ablation best=2%%)")
parser.add_argument("--out_channels_file", default="SFD_channels.txt", type=str)


def main():
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_class = DETECTOR[args.model_name]
    model = model_class().to(device)

    # Import weights trained using only cross-entropy loss.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    loss_fn = SoftNearestNeighborLoss()

    train_dataset = fair_dateset(
        csv_file=os.path.join(args.dataset_dir, "train.csv"),
        owntransforms=data_transforms["train"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    all_feature_map_snnl = {}
    csv_file_path = "SNNL_dic_tmp.csv"
    batch_id = 0

    with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)

        for _, (img_trans, label, attributes) in enumerate(train_dataloader):
            data_dict = {
                "image": img_trans.to(device),
                "label": label.to(device),
                "intersec_label": attributes.to(device),
            }

            with torch.no_grad():
                feature_map = extract_feature_from_512channel_conv(
                    data_dict["image"], model
                )

                feature_map = feature_map.squeeze(0)
                split_features = torch.chunk(feature_map, feature_map.size(1), dim=1)

                feature_map_snnl = []
                for split_feature in split_features:
                    split_feature = split_feature.view(split_feature.shape[0], -1)
                    loss_snnl = (
                        loss_fn(split_feature, data_dict["intersec_label"])
                        .detach()
                        .cpu()
                        .item()
                    )
                    feature_map_snnl.append(loss_snnl)

            all_feature_map_snnl[batch_id] = feature_map_snnl
            csv_writer.writerow(feature_map_snnl)
            batch_id += 1

    array_list = np.array(list(all_feature_map_snnl.values()))
    averaged_array = np.mean(array_list, axis=0)
    data_average = np.array(averaged_array)

    x = np.arange(0, data_average.shape[0])

    n = max(1, round(args.prc / 100.0 * data_average.shape[0]))  # DINGREPRO_LIN: prc% of 512 channels
    threshold = get_min_n(data_average, n)
    print(f"min_{n} ({args.prc}%):", threshold)

    mask = data_average < threshold
    min_n_channels = x[mask]
    print(f"min_{n}_channels:", ", ".join(map(str, min_n_channels)))

    min_n_values = data_average[min_n_channels]
    sorted_indices = np.argsort(min_n_values)
    sorted_min_n_channels = min_n_channels[sorted_indices]
    sorted_min_n_values = min_n_values[sorted_indices]

    print(f"Sorted min_{n}_channels:", sorted_min_n_channels)
    print(f"Sorted min_{n}_values:", sorted_min_n_values)
    with open(args.out_channels_file, "w") as f:
        f.write(" ".join(str(c) for c in sorted_min_n_channels.tolist()))
    print("DINGREPRO_LIN: wrote", len(sorted_min_n_channels), "channels to", args.out_channels_file)


if __name__ == "__main__":
    main()