import sys, torch, pandas as pd, numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transform import fair_df_default_data_transforms as data_transforms
from sklearn.metrics import roc_auc_score, confusion_matrix

def classification_metrics(label, prediction):
    auc = roc_auc_score(label, prediction)
    CM = confusion_matrix(label, prediction >= 0.5)
    TN, FN, TP, FP = CM[0][0], CM[1][0], CM[1][1], CM[0][1]
    fpr = FP/(FP+TN); tpr = TP/(TP+FN)
    return auc, None, fpr, tpr, None, None, None, None, None, None, None, None

ckpt_path, csv_path = sys.argv[1], sys.argv[2]
device = torch.device("cpu")

class DS(Dataset):
    def __init__(self, df, tf):
        self.df, self.tf = df.reset_index(drop=True), tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r['img_path']).convert('RGB')
        return self.tf(img), int(r['label']), int(r['intersec_label'])

df = pd.read_csv(csv_path)
ds = DS(df, data_transforms['test'])
dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=6)

model = torch.load(ckpt_path, map_location=device, weights_only=False)
model.eval()

preds, labels, groups = [], [], []
with torch.no_grad():
    for i, (x, y, g) in enumerate(dl):
        out = model({'image': x, 'label': y})
        p = F.softmax(out['cls'], dim=1)[:, 1].numpy()
        preds.append(p); labels.append(y.numpy()); groups.append(g.numpy())
        print(f"batch {i+1}/{len(dl)}", flush=True)

preds = np.concatenate(preds); labels = np.concatenate(labels); groups = np.concatenate(groups)
np.savez(sys.argv[3] if len(sys.argv) > 3 else 'eval_raw.npz', preds=preds, labels=labels, groups=groups)
GNAMES = {0:'male-asian',1:'male-white',2:'male-black',3:'male-others',
          4:'nonmale-asian',5:'nonmale-white',6:'nonmale-black',7:'nonmale-others'}

print(f"\nOverall AUC: {classification_metrics(labels, preds)[0]:.4f}  n={len(labels)}")
for g in sorted(set(groups)):
    m = groups == g
    n = m.sum()
    if n < 10 or len(set(labels[m])) < 2:
        print(f"  {GNAMES.get(g,g):14s} n={n:6d}  NE (too few / single-class)")
        continue
    auc, er, fpr, tpr, acc, prec, fg, eer, pr, nr, fnr, tnr = classification_metrics(labels[m], preds[m])
    print(f"  {GNAMES.get(g,g):14s} n={n:6d}  AUC={auc:.4f}  FPR={fpr:.4f}  TPR={tpr:.4f}")
