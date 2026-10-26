"""Audit (21 Oct): score a Ding checkpoint on FF++ validation and the four test sets, saving the raw logit
difference (cls[:,1] - cls[:,0]) next to the softmax probability, so a validation-fitted intercept can be applied
exactly as for CFES (scripts/calibrate.py). Same transforms, CSVs and model loading as eval_ding_protocol.py.

  python ding_score_logits.py --ckpt <pth> --tag full_s7_ep49 --outdir ../../results/audit_scores
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eval_ding_protocol import CSV, DATA, DS, load_model

SETS = {'ffval': [f'{DATA}/ff++_ding/realval.csv', f'{DATA}/ff++_ding/fakeval.csv'],
        'ff++': [CSV['ff++']], 'dfdc': [CSV['dfdc']], 'celebdf': [CSV['celebdf']], 'dfd': [CSV['dfd']]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--sets', default='ffval,ff++,dfdc,celebdf,dfd')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.outdir, exist_ok=True)
    model = load_model(a.ckpt, dev)
    for s in a.sets.split(','):
        out = f'{a.outdir}/{a.tag}_{s}.npz'
        if os.path.exists(out):
            continue
        df = pd.concat([pd.read_csv(c) for c in SETS[s]], ignore_index=True)
        lg, pr = [], []
        with torch.no_grad():
            for x, y, _ in DataLoader(DS(df), batch_size=128, num_workers=8):
                c = model({'image': x.to(dev), 'label': y.to(dev)})['cls'].float()
                lg.append((c[:, 1] - c[:, 0]).cpu().numpy()); pr.append(F.softmax(c, dim=1)[:, 1].cpu().numpy())
        np.savez(out, logit=np.concatenate(lg), prob=np.concatenate(pr), label=df.label.values,
                 group=df.intersec_label.values, img_path=df.img_path.values.astype(str))
        print(a.tag, s, len(df), 'device', dev, flush=True)


if __name__ == '__main__':
    main()
