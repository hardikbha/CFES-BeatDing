"""FairForge-2 in-domain table (pre-registration decisions/2026-10-22-1920-fairness-and-ff2-plan.md, section D):
score a Ding checkpoint on FF2 TEST (test.csv, 182,884) and on the FF2 select split (realval.csv + fakeval.csv,
74,974; used for the validation-chosen threshold). Same model loading and test transform as eval_ding_protocol.py
(fair_df_default_data_transforms['test']) and the same raw-logit definition as ding_score_logits.py
(cls[:,1] - cls[:,0]). Writes <out_dir>/<prefix><set>.npz with logit, prob, label, group, img_path.

  python ding_score_ff2.py --ckpt <pth> --out_dir <run>/scores [--prefix base_] [--sets ff2_test,ff2_select]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eval_ding_protocol import DS, load_model

FF2 = '/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/data_ding_schema/fairforge2_full'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--prefix', default='')
    ap.add_argument('--sets', default='ff2_test,ff2_select')
    ap.add_argument('--data_dir', default=FF2, help='Ding-schema dir holding test.csv, realval.csv, fakeval.csv')
    ap.add_argument('--max_items', type=int, default=None, help='smoke only: first N rows of each set')
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    sets = {'ff2_test': ['test.csv'], 'ff2_select': ['realval.csv', 'fakeval.csv']}
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out_dir, exist_ok=True)
    model = None
    for s in a.sets.split(','):
        out = os.path.join(a.out_dir, f'{a.prefix}{s}.npz')
        if os.path.exists(out):
            print('exists', out, flush=True)
            continue
        if model is None:
            model = load_model(a.ckpt, dev)
        df = pd.concat([pd.read_csv(os.path.join(a.data_dir, c)) for c in sets[s]], ignore_index=True)
        if a.max_items:
            df = df.iloc[:a.max_items]
        lg, pr = [], []
        with torch.no_grad():
            for x, y, _ in DataLoader(DS(df), batch_size=a.batch_size, num_workers=a.workers):
                c = model({'image': x.to(dev), 'label': y.to(dev)})['cls'].float()
                lg.append((c[:, 1] - c[:, 0]).cpu().numpy())
                pr.append(F.softmax(c, dim=1)[:, 1].cpu().numpy())
        tmp = out[:-4] + '.tmp.npz'
        np.savez(tmp, logit=np.concatenate(lg), prob=np.concatenate(pr), label=df.label.values,
                 group=df.intersec_label.values, img_path=np.asarray(df.img_path.tolist(), dtype='U'),  # plain
                 # unicode, so np.load works without allow_pickle (pandas 3 string columns become object arrays)
                 ckpt=np.array(os.path.abspath(a.ckpt)))
        os.replace(tmp, out)
        print(a.prefix + s, len(df), 'device', dev, '->', out, flush=True)


if __name__ == '__main__':
    main()
