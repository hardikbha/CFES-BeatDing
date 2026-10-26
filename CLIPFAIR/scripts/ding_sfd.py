"""Ding et al. CVPR'26 Sensitive-Feature Decoupling (SFD) on the CLIP backbone: DM SFD.py ported.

Official (agents/DINGREPRO_LIN/code/Ding_mod/SFD.py): the CE-trained base detector (here: CFES runs/clip_s<seed>_p1,
backbone; it is also the start checkpoint of every CLIPFAIR baseline), one pass over the WHOLE training set
(train.csv, batch 64, shuffled, drop_last False; SFD.py:116-127); for every batch and every channel c of the
512-channel adjust_channel conv output (SFD.py:61-81, 144-149), the soft-nearest-neighbour loss of the channel's
spatial map (SFD.py:150-160) w.r.t. the intersectional group label (cos distance, T = 1); average over batches; the
prc = 2% channels with the LOWEST average SNNL (most clustered by demographic group) are decoupled (SFD.py:166-188).
CLIP adaptation: channels = the 1024 ln_post channels; a channel's "spatial map" = its values on the 16 x 16 patch
tokens (256-d); n = round(2% x 1024) = 20. GDA then zeros those channels of the CLS feature fed to the classifier
(cfes/clipfair_models.DingDetector; official: out-channel pruning of the adjust_channel conv, GDA.py:63-74,157-158).

  python scripts/ding_sfd.py --config configs/ffpp/ding.yaml --out runs/ffpp_ding_sfd_s7 --set seed=7 \
         model.init_backbone_ckpt=<CFES runs/clip_s7_p1/last.pt>
Writes <out>/sfd_channels.json (channels, values, threshold, n) and <out>/snnl_per_channel.npy.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cfes.clipfair_losses import select_sfd_channels, snnl_per_channel  # noqa: E402
from cfes.clipfair_models import build_model  # noqa: E402
from cfes.config import dump_config, load_config  # noqa: E402
from cfes.data import ImageTable, read_split  # noqa: E402
from cfes.utils import set_seed  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, list(a.set) + ["clipfair.ding.channels_json=null"])
    dc = cfg["clipfair"]["ding"]
    os.makedirs(a.out, exist_ok=True)
    dump_config(cfg, os.path.join(a.out, "config.yaml"))
    set_seed(cfg["seed"])
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg, device).eval()
    d = cfg["data"]
    df = read_split({**d["train"], "max_items": cfg["debug"].get("max_train_items")})
    ds = ImageTable(df, d["input_size"], d.get("resample", "bicubic"))
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    dl = DataLoader(ds, batch_size=int(dc["sfd_batch_size"]), shuffle=True, num_workers=d["num_workers"],
                    pin_memory=True, drop_last=False, generator=g)
    amp = bool(cfg["optim"]["amp"]) and device.type == "cuda"
    acc, nb, t0 = None, 0, time.time()
    for i, b in enumerate(dl):
        if dc.get("sfd_max_batches") and i >= int(dc["sfd_max_batches"]):
            break
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            feat, _ = model.backbone(b["image"].to(device, non_blocking=True))       # (B,G,G,C)
        B, G, _, C = feat.shape
        fm = feat.float().reshape(B, G * G, C).permute(0, 2, 1)                       # (B,C,P): channel maps
        v = snnl_per_channel(fm, b["group"].to(device), float(dc.get("snnl_temperature", 1.0))).double().cpu()
        acc = v if acc is None else acc + v
        nb += 1
        if i % 100 == 0:
            print(f"sfd batch {i}/{len(dl)} {time.time() - t0:.0f}s", flush=True)
    avg = (acc / nb).numpy()
    if not np.isfinite(avg).all():
        print(f"WARNING: {int((~np.isfinite(avg)).sum())} channels with non-finite average SNNL", flush=True)
    ch, thr, n = select_sfd_channels(avg, float(dc["prc"]))
    np.save(os.path.join(a.out, "snnl_per_channel.npy"), avg)
    rep = {"channels": ch, "values": [float(avg[c]) for c in ch], "threshold": thr, "n": n, "prc": float(dc["prc"]),
           "n_channels": int(avg.shape[0]), "n_batches": nb, "batch_size": int(dc["sfd_batch_size"]),
           "n_images": int(len(ds)) if not dc.get("sfd_max_batches") else None, "seed": int(cfg["seed"]),
           "base": cfg["model"].get("init_backbone_ckpt"), "feature": "CLIP ln_post patch tokens (16x16) per channel",
           "snnl_mean_all": float(np.nanmean(avg)), "seconds": time.time() - t0}
    tmp = os.path.join(a.out, "sfd_channels.json.tmp")
    json.dump(rep, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(a.out, "sfd_channels.json"))
    print(f"SFD: {n} channels {ch} (threshold {thr:.5f}, mean SNNL {np.nanmean(avg):.5f}, {nb} batches)", flush=True)


if __name__ == "__main__":
    main()
