# Aggregate analysis of low-level statistics (TRAIN/VAL only). Prints aggregates; writes stats_summary.json.
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
OUT = '/home/sachin.chaudhary/ces_fl_runs/ff2_source_diag_20261106'
f = pd.read_pickle(OUT + '/_features_PRIVATE.pkl')
S = {}
print('q_hash', f.groupby(['split', 'label', 'src']).q_hash.agg(lambda s: dict(s.value_counts())).to_dict())
print('sampling', f.groupby(['split', 'label', 'src']).sampling.agg(lambda s: dict(s.value_counts())).to_dict())
S['jpeg_tables'] = {str(k): v for k, v in f.groupby(['split', 'label', 'src']).q_hash.agg(lambda s: dict(s.value_counts())).to_dict().items()}
S['sampling'] = {str(k): v for k, v in f.groupby(['split', 'label', 'src']).sampling.agg(lambda s: {int(a): int(b) for a, b in s.value_counts().items()}).to_dict().items()}
SC = ['bytes', 'log_lap', 'noise', 'hf_y', 'hf_c', 'block8', 'sat', 'val', 'y_std', 'dark', 'flat8']
v = f[f.split == 'val']
grp = v.groupby(['label', 'src'])[SC]
med = grp.median(); q25 = grp.quantile(.25); q75 = grp.quantile(.75)
print('VAL medians\n', med.round(3).to_string())
S['val_median'] = {f'{a}|{b}': r.round(4).to_dict() for (a, b), r in med.iterrows()}
S['val_q25'] = {f'{a}|{b}': r.round(4).to_dict() for (a, b), r in q25.iterrows()}
S['val_q75'] = {f'{a}|{b}': r.round(4).to_dict() for (a, b), r in q75.iterrows()}
# per-method medians of key stats for VAL fakes, by source
pm = v[v.label == 1].groupby(['method', 'src'])[['log_lap', 'noise', 'hf_y', 'bytes']].median().unstack()
print('VAL fakes per method\n', pm.round(2).to_string())
S['val_fake_method_median'] = json.loads(pm.round(3).to_json())
# fraction of each VAL group whose sharpness exceeds the FairFace-real p95 / falls below FFHQ-real p5
ffr = v[(v.label == 0) & (v.src == 'img')]; fqr = v[(v.label == 0) & (v.src == 'ffhq')]
for k in ['log_lap', 'hf_y', 'noise']:
    t95 = ffr[k].quantile(.95); t05 = fqr[k].quantile(.05)
    S[f'frac_above_fairface_real_p95_{k}'] = {f'{a}|{b}': float((g[k] > t95).mean()) for (a, b), g in v.groupby(['label', 'src'])}
    print(k, 'frac > FairFace-real p95:', S[f'frac_above_fairface_real_p95_{k}'])
FEAT = [c for c in f.columns if c in SC or c.startswith(('ry', 'rc'))]
tr = f[f.split == 'train']
# (B) source-of-real classifier: FFHQ real vs FairFace real, fit on TRAIN reals
r = tr[tr.label == 0]
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced'))
clf.fit(r[FEAT], (r.src == 'ffhq').astype(int))
vr = v[v.label == 0]
p = clf.predict_proba(vr[FEAT])[:, 1]
S['source_clf_val_real_auc'] = float(roc_auc_score((vr.src == 'ffhq').astype(int), p))
pv = clf.predict_proba(v[FEAT])[:, 1]; v = v.assign(p_ffhq_like=pv)
S['source_clf_frac_ffhq_like'] = {f'{a}|{b}': float((g.p_ffhq_like > .5).mean()) for (a, b), g in v.groupby(['label', 'src'])}
print('source clf VAL real AUC', S['source_clf_val_real_auc'], 'frac FFHQ-real-like', S['source_clf_frac_ffhq_like'])
fm = v[v.label == 1].groupby(['method', 'src']).p_ffhq_like.apply(lambda s: (s > .5).mean()).unstack()
S['source_clf_frac_ffhq_like_by_method'] = json.loads(fm.round(3).to_json())
print(fm.round(2).to_string())
# (C) real-vs-fake classifier on statistics only, fit on TRAIN sample, thresholded at 10% VAL-real FPR
for name, model in [('logreg', make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
                    ('hgb', HistGradientBoostingClassifier(max_iter=300, random_state=0))]:
    model.fit(tr[FEAT], tr.label)
    s = model.predict_proba(v[FEAT])[:, 1]
    real = s[v.label.values == 0]; thr = np.sort(real)[::-1][int(0.1 * len(real))]
    res = {'val_auc_all': float(roc_auc_score(v.label, s))}
    for src in ['ffhq', 'img']:
        m = (v.src.values == src)
        res[f'fpr_{src}'] = float((s[m & (v.label.values == 0)] > thr).mean())
        res[f'tpr_{src}_fakes'] = float((s[m & (v.label.values == 1)] > thr).mean())
        res[f'auc_within_{src}'] = float(roc_auc_score(v.label.values[m], s[m]))
    S[f'stat_detector_{name}'] = res; print(name, res)
# (D) relation with L_A logits on VAL reals
d = json.load(open('/home/sachin.chaudhary/ces_fl_runs/ff2_lh_ps_20261106_r1/L_A/validation_scores_PRIVATE.json'))
z = pd.Series(np.asarray(d['logits'], float), index=[r['image'] for r in d['rows']]); thr = float(d['threshold'])
vr = v[v.label == 0].copy(); vr['z'] = z.reindex(vr.image).values; vr['flag'] = vr.z >= thr
from scipy.stats import spearmanr
S['LA_spearman_val_reals'] = {}
for src in ['ffhq', 'img', 'all']:
    g = vr if src == 'all' else vr[vr.src == src]
    S['LA_spearman_val_reals'][src] = {k: float(spearmanr(g[k], g.z).correlation) for k in ['log_lap', 'hf_y', 'noise', 'bytes', 'p_ffhq_like']}
print('spearman', S['LA_spearman_val_reals'])
# FairFace reals: L_A FPR by sharpness quintile (dose-response)
g = vr[vr.src == 'img'].copy(); g['q'] = pd.qcut(g.log_lap, 5, labels=False)
S['LA_fpr_fairface_real_by_laplacian_quintile'] = g.groupby('q').flag.agg(['size', 'mean']).round(4).to_dict('index')
g['qh'] = pd.qcut(g.hf_y, 5, labels=False)
S['LA_fpr_fairface_real_by_hf_quintile'] = g.groupby('qh').flag.agg(['size', 'mean']).round(4).to_dict('index')
h = vr[vr.src == 'ffhq'].copy(); h['q'] = pd.qcut(h.log_lap, 3, labels=False)
S['LA_fpr_ffhq_real_by_laplacian_tercile'] = h.groupby('q').flag.agg(['size', 'mean']).round(4).to_dict('index')
# FairFace reals that the source classifier calls FFHQ-like vs not
S['LA_fpr_fairface_real_by_ffhq_like'] = vr[vr.src == 'img'].groupby(vr.p_ffhq_like > .5).flag.agg(['size', 'mean']).round(4).to_dict('index')
S['LA_fpr_by_src'] = vr.groupby('src').flag.agg(['size', 'mean']).round(4).to_dict('index')
print(json.dumps({k: S[k] for k in S if k.startswith('LA_fpr')}, indent=0, default=str))
json.dump(S, open(OUT + '/stats_summary.json', 'w'), indent=1, default=str)
