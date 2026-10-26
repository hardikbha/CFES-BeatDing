"""Does the released GDA fairness loss reach the parameters? (review 4, item 1)

Runs the EXACT training step of the released GDA.py (repo ywh1093/Fairness-Optimization @ 059ec37) on real FF++ data:
same XceptionDetector, same base checkpoint (DING_base_s7/49.pth), same SFD channel mask, same frozen parameters
(only backbone.last_linear trains), same SGD(lr 1e-3, momentum 0.9, wd 5e-3), same data classes and transforms, and the
released loss block verbatim (Sinkhorn fairness loss and total loss computed inside `with torch.no_grad():`).

Reported for the released block:
  * the autograd graph of total_loss (grad_fn chain) and whether fair_loss requires grad;
  * the gradient of lambda * fair_loss alone w.r.t. last_linear (norm);
  * one optimizer step from identical states with lambda = 0, 0.005 (paper) and 1000: max |param difference|.
The same three measurements are repeated for a SEPARATELY LABELLED variant with the fairness term connected (computed
outside no_grad). Nothing in the official reproduction is modified.

  python verify_gda_grad.py --out ../../results/GDA_GRAD_CHECK.json      (qwen38 env, CPU is enough)
"""
import argparse
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn.utils.prune as prune
import torch.optim as optim
from geomloss import SamplesLoss
from torch.utils.data import DataLoader

from detectors import DETECTOR
from src.dataprocess import (CombinedDataset, MergedDataset, create_group_dataloaders, fair_train_dateset,
                             find_last_adjust_channel_conv_layer)
from transform import resnet_default_data_transforms as data_transforms
from GDA import OutChannelPruning

A = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN"


def graph_names(t, depth=6):
    out, frontier = [], [t.grad_fn] if t.grad_fn is not None else []
    for _ in range(depth):
        nxt = []
        for f in frontier:
            out.append(type(f).__name__)
            nxt += [g for g, _ in f.next_functions if g is not None]
        frontier = nxt
    return out


def build(seed_ckpt, channels):
    model = DETECTOR["XceptionDetector"]()
    model.load_state_dict(torch.load(seed_ckpt, map_location="cpu", weights_only=True), strict=True)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.backbone.last_linear.parameters():
        p.requires_grad = True
    layer = find_last_adjust_channel_conv_layer(model)
    prune.custom_from_mask(layer, name="weight", mask=OutChannelPruning(channels).compute_mask(layer.weight))
    model.train()
    return model


def step(model, combined, group, lam, connected):
    """One optimizer step; returns (param after step, grad of lam*fair alone, fair value, total graph, fair requires_grad)."""
    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-03)
    fairLoss = SamplesLoss(loss="sinkhorn", p=2, blur=5e-4)
    preds = model(combined)
    main_loss = model.get_losses(combined, preds)["overall"]
    if not connected:                                   # released code, verbatim
        with torch.no_grad():
            group_preds = model(group)
            fair_loss = fairLoss(preds["cls"], group_preds["cls"])
            total_loss = main_loss + lam * fair_loss
    else:                                               # separately labelled variant: fairness term connected
        group_preds = model(group)
        fair_loss = fairLoss(preds["cls"], group_preds["cls"])
        total_loss = main_loss + lam * fair_loss
    info = {"fair_loss": float(fair_loss), "fair_requires_grad": bool(fair_loss.requires_grad),
            "total_requires_grad": bool(total_loss.requires_grad), "total_graph": graph_names(total_loss)}
    W = model.backbone.last_linear.weight
    if fair_loss.requires_grad:
        g = torch.autograd.grad(lam * fair_loss, W, retain_graph=True, allow_unused=True)[0]
        info["fair_grad_norm"] = 0.0 if g is None else float(g.norm())
    else:
        info["fair_grad_norm"] = 0.0
    opt.zero_grad()
    total_loss.backward()
    info["total_grad_norm"] = float(W.grad.norm())
    opt.step()
    return {k: v.detach().clone() for k, v in model.backbone.last_linear.state_dict().items()}, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=f"{A}/results/GDA_GRAD_CHECK.json")
    a = ap.parse_args()
    torch.manual_seed(121); np.random.seed(121); random.seed(121)       # GDA.py always uses seed 121
    ds = f"{A}/data/ff++_ding"
    channels = [int(x) for x in open(f"{A}/results/DING_full_s{a.seed}/SFD_channels.txt").read().split()]
    real = fair_train_dateset(dataset_dir=ds, owntransforms=data_transforms["train"], csv_file=f"{ds}/realtrain.csv")
    fake = fair_train_dateset(dataset_dir=ds, owntransforms=data_transforms["train"], csv_file=f"{ds}/faketrain.csv")
    (real_data, fake_data) = next(iter(DataLoader(CombinedDataset(real, fake), batch_size=32, shuffle=True, num_workers=4)))
    combined = {"image": torch.cat([real_data[0], fake_data[0]]), "label": torch.cat([real_data[1], fake_data[1]]),
                "intersec_label": torch.cat([real_data[2], fake_data[2]])}
    rg, _ = create_group_dataloaders(dataset_dir=ds, csv_file=f"{ds}/realtrain.csv", data_transforms=data_transforms,
                                     batch_size=32, num_workers=4)
    fg, _ = create_group_dataloaders(dataset_dir=ds, csv_file=f"{ds}/faketrain.csv", data_transforms=data_transforms,
                                     batch_size=32, num_workers=4)
    rb, fb = next(iter(DataLoader(MergedDataset(rg[3], fg[3]), batch_size=32, shuffle=True, num_workers=4)))
    group = {"image": torch.cat([rb[0], fb[0]]), "label": torch.cat([rb[1], fb[1]]), "intersec_label": torch.cat([rb[2], fb[2]])}
    base = build(f"{A}/results/DING_base_s{a.seed}/49.pth", channels)
    rep = {"repo": "https://github.com/ywh1093/Fairness-Optimization @ 059ec37fc969825bab9dc5f79e01c9864d00eaec",
           "base_ckpt": f"DING_base_s{a.seed}/49.pth", "channels": channels, "batch": "32 real + 32 fake; group batch = group 3",
           "optimizer": "SGD lr 1e-3 momentum 0.9 wd 5e-3 (as GDA.py)", "released": {}, "connected": {}}
    for mode, connected in (("released", False), ("connected", True)):
        params = {}
        for lam in (0.0, 0.005, 1000.0):
            torch.manual_seed(0)
            p, info = step(copy.deepcopy(base), combined, group, lam, connected)
            params[lam] = p
            rep[mode][f"lambda={lam}"] = info
        for lam in (0.005, 1000.0):
            rep[mode][f"max|param(lambda={lam}) - param(lambda=0)| after one step"] = max(
                float((params[lam][k] - params[0.0][k]).abs().max()) for k in params[0.0])
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
