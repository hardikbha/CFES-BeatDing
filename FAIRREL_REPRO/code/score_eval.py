"""FAIRREL_REPRO per-image scorer. Derived from L_fairrel_protocol/code/l_fairrel_eval.py (pointing game removed;
paths saved; FF++-val penultimate features saved for the demographic probe).

Metric path identical to the official test.py: FairReL's own DeepfakeDataset(train=False) -> model.predict (fp32)
-> threshold_at_tpr(FF++ val, cfg tpr_anchor = 0.90) -> fairness_report per test set. run_task.sh also runs the
official test.py on the same checkpoint and asserts equality.

  --checkpoint best.pt   : trained model
  --init                 : no checkpoint; the untrained FairReLDetector == frozen OpenAI CLIP ViT-L/14 (the SVD split
                           W = W_p + U_r S_r V_r^T is exact at init); only val features are written (no scoring).
Val features: z = CLS post-LN (the classifier input, 1024-d) and z_delta = residual-only path (SLMA's input).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

FR = "/home/sachin.chaudhary/ICLR_2027/related_work/FairReL"
sys.path.insert(0, FR)
from fairrel.data import DeepfakeDataset                                      # noqa: E402
from fairrel.metrics import fairness_report, format_report, threshold_at_tpr  # noqa: E402
from fairrel.models import FairReLDetector                                    # noqa: E402
from fairrel.utils import load_config                                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=f"{FR}/configs/fairrel_clip_vitl14.yaml")
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--init", action="store_true")
ap.add_argument("--val-csv", required=True)
ap.add_argument("--test", nargs="*", default=[], metavar="NAME=CSV")
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--num-workers", type=int, default=8)
ap.add_argument("--output", default=None)
ap.add_argument("--save-scores", default=None)
ap.add_argument("--save-feats", required=True)
a = ap.parse_args()
assert a.init != (a.checkpoint is not None), "exactly one of --init / --checkpoint"

cfg = load_config(a.config)
dev = torch.device("cuda")
m = cfg["model"]
torch.manual_seed(0)
model = FairReLDetector(m["backbone"], m["pretrained"], m["svd_rank"], m.get("pooling", "cls")).to(dev)
st = {}
if a.checkpoint:
    st = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    model.load_state_dict(st["model"] if "model" in st else st)
model.eval()


@torch.no_grad()
def run(csv, feats=False):
    ds = DeepfakeDataset(csv, "", cfg["train"]["input_size"], train=False)
    S, Y, G, Z, ZD = [], [], [], [], []
    for img, y, g in DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                                pin_memory=True):
        img = img.to(dev, non_blocking=True)
        _, z, logit = model(img)
        S += torch.sigmoid(logit).float().cpu().tolist()          # == model.predict
        Y += y.tolist(); G += g.tolist()
        if feats:
            Z.append(z.float().cpu().numpy())
            ZD.append(model.forward_residual(img).float().cpu().numpy())
    out = dict(s=np.asarray(S, np.float64), y=np.asarray(Y), g=np.asarray(G), paths=np.asarray(ds.paths))
    if feats:
        out["z"], out["zd"] = np.concatenate(Z), np.concatenate(ZD)
    return out


v = run(a.val_csv, feats=True)
np.savez(a.save_feats + ".tmp.npz", z=v["z"].astype(np.float32), zd=v["zd"].astype(np.float32),
         y=v["y"], g=v["g"], path=v["paths"], score=v["s"].astype(np.float32))
os.replace(a.save_feats + ".tmp.npz", a.save_feats)
print(f"saved val features {v['z'].shape} -> {a.save_feats}")
if a.init:
    sys.exit(0)

thr = threshold_at_tpr(v["s"], v["y"], cfg["eval"]["tpr_anchor"])
print(f"operating point: threshold {thr:.6f}")
res = {"threshold": thr, "checkpoint": os.path.abspath(a.checkpoint), "checkpoint_epoch": st.get("epoch"),
       "val": fairness_report(v["s"], v["y"], v["g"], thr)}
saved = {"val": v["s"], "val_y": v["y"], "val_g": v["g"], "val_path": v["paths"]}
for spec in a.test:
    name, _, csv = spec.partition("=")
    t = run(csv)
    res[name] = fairness_report(t["s"], t["y"], t["g"], thr)
    saved.update({name: t["s"], name + "_y": t["y"], name + "_g": t["g"], name + "_path": t["paths"]})
    print(format_report(name, res[name]))
json.dump(res, open(a.output + ".tmp", "w"), indent=1, default=float)
os.replace(a.output + ".tmp", a.output)
np.savez(a.save_scores + ".tmp.npz", **saved)
os.replace(a.save_scores + ".tmp.npz", a.save_scores)
print("saved", a.output, a.save_scores)
