"""Score one checkpoint on Ding et al.'s test sets with the paper's Eq. 8-10 metrics.

Subgroups are read from the CSV's `intersec_label` column by name: the released
test.py reads row[4], which is iswhite/isblack in every released CSV.
"""
import argparse, json, os
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from transform import fair_df_default_data_transforms as data_transforms
from detectors import DETECTOR

DATA = '/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/data'
CSV = {'ff++': f'{DATA}/ff++/test.csv', 'dfdc': f'{DATA}/dfdc/test.csv',
       'celebdf': f'{DATA}/celebdf/test.csv', 'dfd': f'{DATA}/dfd/test.csv',
       # FairForge-2 TEST (our dataset, Ding label schema; added 18 Oct for the 'FF2 as a test benchmark' table)
       'ff2': '/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/data_ding_schema/fairforge2/test.csv',
       # FULL FairForge-2 TEST (837,220-image build of 18 Oct; 182,884 test images)
       'ff2full': '/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/data_ding_schema/fairforge2_full/test.csv'}
EIGHT = {0: ('male', 'asian'), 1: ('male', 'white'), 2: ('male', 'black'), 3: ('male', 'others'),
         4: ('nonmale', 'asian'), 5: ('nonmale', 'white'), 6: ('nonmale', 'black'), 7: ('nonmale', 'others')}
SIX = {0: ('male', 'white'), 1: ('male', 'black'), 2: ('male', 'others'),
       3: ('nonmale', 'white'), 4: ('nonmale', 'black'), 5: ('nonmale', 'others')}
SCHEME = {'ff++': EIGHT, 'dfdc': EIGHT, 'celebdf': SIX, 'dfd': SIX, 'ff2': EIGHT, 'ff2full': EIGHT}


class DS(Dataset):
    def __init__(self, df):
        self.p, self.y, self.g = df.img_path.values, df.label.values, df.intersec_label.values
        self.tf = data_transforms['test']

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert('RGB')), int(self.y[i]), int(self.g[i])


def load_model(path, device):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        model = DETECTOR['XceptionDetector']()
        model.load_state_dict(obj, strict=True)
    else:
        model = obj
    return model.to(device).eval()


def group_stats(y, p):
    if len(y) == 0 or len(set(y)) < 2:
        return None
    TN, FP, FN, TP = confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel()
    n = len(y)
    return dict(n=int(n), auc=float(roc_auc_score(y, p)), FPR=float(FP / (FP + TN)), TPR=float(TP / (TP + FN)),
                PR=float((TP + FP) / n), NR=float((TN + FN) / n))


def ding_metrics(y, p, g, scheme):
    overall = group_stats(y, p)
    oauc, ofpr = overall['auc'], overall['FPR']
    partitions = {
        'gender': {s: [k for k, v in scheme.items() if v[0] == s] for s in ['male', 'nonmale']},
        'race': {r: [k for k, v in scheme.items() if v[1] == r] for r in ['asian', 'white', 'black', 'others']},
        'intersection': {f'{v[0]},{v[1]}': [k] for k, v in scheme.items()},
    }
    out = {'overall_auc': 100 * oauc, 'n': overall['n'], 'groups': {}}
    for level, part in partitions.items():
        stats = {}
        for name, ids in part.items():
            s = group_stats(y[np.isin(g, ids)], p[np.isin(g, ids)]) if ids else None
            if s:
                stats[name] = s
        out['groups'][level] = stats
        v = list(stats.values())
        out[level] = {
            'F_FPR': 100 * sum(abs(s['FPR'] - ofpr) for s in v),
            'F_DP': 100 * max(max(s['PR'] for s in v) - min(s['PR'] for s in v),
                              max(s['NR'] for s in v) - min(s['NR'] for s in v)),
            'es_AUC': 100 * oauc / (1 + sum(abs(s['auc'] - oauc) for s in v)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--datasets', default='ff++,dfdc,celebdf,dfd')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.outdir, exist_ok=True)
    model = load_model(a.ckpt, device)
    for ds in a.datasets.split(','):
        df = pd.read_csv(CSV[ds])
        preds = []
        with torch.no_grad():
            for x, y, _ in DataLoader(DS(df), batch_size=a.batch_size, num_workers=a.workers):
                out = model({'image': x.to(device), 'label': y.to(device)})
                preds.append(F.softmax(out['cls'], dim=1)[:, 1].cpu().numpy())
        p, y, g = np.concatenate(preds), df.label.values, df.intersec_label.values
        np.savez(f'{a.outdir}/{a.tag}_{ds}.npz', preds=p, labels=y, groups=g)
        m = ding_metrics(y, p, g, SCHEME[ds])
        m.update(ckpt=a.ckpt, dataset=ds, device=str(device))
        with open(f'{a.outdir}/{a.tag}_{ds}.json', 'w') as f:
            json.dump(m, f, indent=1)
        print(f"{a.tag} {ds}: AUC {m['overall_auc']:.2f} | gender {m['gender']} | race {m['race']} | inter {m['intersection']}", flush=True)


if __name__ == '__main__':
    main()
