import sys, numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

d = np.load(sys.argv[1])
preds, labels, groups = d['preds'], d['labels'], d['groups']

GENDER_OF = {0:'male',1:'male',2:'male',3:'male',4:'nonmale',5:'nonmale',6:'nonmale',7:'nonmale'}
RACE_OF   = {0:'asian',1:'white',2:'black',3:'others',4:'asian',5:'white',6:'black',7:'others'}
INTER_NAMES = {0:'male,asian',1:'male,white',2:'male,black',3:'male,others',
               4:'nonmale,asian',5:'nonmale,white',6:'nonmale,black',7:'nonmale,others'}

def stats(mask):
    y, p = labels[mask], preds[mask]
    n = len(y)
    if n == 0 or len(set(y)) < 2:
        return None
    auc = roc_auc_score(y, p)
    CM = confusion_matrix(y, p >= 0.5)
    TN, FN, TP, FP = CM[0][0], CM[1][0], CM[1][1], CM[0][1]
    FPR = FP/(FP+TN); TPR = TP/(TP+FN)
    PR = (TP+FP)/n; NR = (TN+FN)/n
    return dict(n=n, auc=auc, FPR=FPR, TPR=TPR, PR=PR, NR=NR)

overall = stats(np.ones(len(labels), dtype=bool))
overall_auc = overall['auc']

def es_auc_fdp_fefpr(submetrics, overall_fpr_full):
    aucs = [m['auc'] for m in submetrics]
    fprs = [m['FPR'] for m in submetrics]
    prs  = [m['PR'] for m in submetrics]
    nrs  = [m['NR'] for m in submetrics]
    tmp = sum(abs(a - overall_auc) for a in aucs)
    es_auc = overall_auc / (tmp + 1)
    f_dp = 100 * max(max(prs)-min(prs), max(nrs)-min(nrs))
    f_efpr = 100 * sum(abs(f - overall_fpr_full) for f in fprs)
    f_fpr_range = 100 * (max(fprs) - min(fprs))
    return es_auc*100, f_dp, f_efpr, f_fpr_range

# Gender (g1)
gender_groups = {}
for g in ['male','nonmale']:
    mask = np.isin(groups, [k for k,v in GENDER_OF.items() if v==g])
    gender_groups[g] = stats(mask)
g1_list = [v for v in gender_groups.values() if v]
es_g1, fdp_g1, fefpr_g1, frange_g1 = es_auc_fdp_fefpr(g1_list, overall['FPR'])

# Race (g2)
race_groups = {}
for r in ['asian','white','black','others']:
    mask = np.isin(groups, [k for k,v in RACE_OF.items() if v==r])
    race_groups[r] = stats(mask)
g2_list = [v for v in race_groups.values() if v]
es_g2, fdp_g2, fefpr_g2, frange_g2 = es_auc_fdp_fefpr(g2_list, overall['FPR'])

# Intersection
inter_groups = {}
for gi in range(8):
    inter_groups[INTER_NAMES[gi]] = stats(groups == gi)
inter_list = [v for v in inter_groups.values() if v]
es_inter, fdp_inter, fefpr_inter, frange_inter = es_auc_fdp_fefpr(inter_list, overall['FPR'])

print(f"\n=== Ding-schema fairness table (dataset=FF++, our reproduction) ===")
print(f"Overall AUC: {overall_auc*100:.2f}  n={overall['n']}\n")
print(f"{'':12s} {'F_FPR':>8s} {'F_DP':>8s} {'es-AUC':>8s}")
print(f"{'Gender':12s} {fefpr_g1:8.2f} {fdp_g1:8.2f} {es_g1:8.2f}")
print(f"{'Race':12s} {fefpr_g2:8.2f} {fdp_g2:8.2f} {es_g2:8.2f}")
print(f"{'Intersect':12s} {fefpr_inter:8.2f} {fdp_inter:8.2f} {es_inter:8.2f}")
print(f"\nPer-subgroup raw: ")
for name, m in {**gender_groups, **race_groups, **inter_groups}.items():
    if m:
        print(f"  {name:16s} n={m['n']:6d} AUC={m['auc']*100:6.2f} FPR={m['FPR']*100:6.2f} TPR={m['TPR']*100:6.2f}")
