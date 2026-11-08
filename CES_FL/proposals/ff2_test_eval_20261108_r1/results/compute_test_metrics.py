# Computes shared downstream metrics on the FF2 public TEST split for the 10 frozen checkpoints scored by the
# export_*.py scripts in this directory. The group definitions and the `compute()` function are copied verbatim
# from ~/ces_fl_runs/ff2_fairness_tables_20261105/aggregate_rfs.py (the exact code already used and verified for
# every VAL table in this project) -- not reimplemented -- so TEST numbers are computed the same way VAL numbers
# were. Every score file is checked to be bound to the canonical TEST manifest and have the full 87,564 rows before
# any metric is computed.
import csv, json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

OUT = Path('/home/sachin.chaudhary/ces_fl_runs/ff2_test_eval_20261108/metrics')
OUT.mkdir(parents=True, exist_ok=True)
TEST_SHA = 'dd5eb9ef070f658fe6b7f744c47f73e25ce7c53f948bc272f4f16c69a37d4099'
N_TEST = 87564
MIN_SUPPORT = 20

SPECS = {  # model -> list of (seed_label, test_scores_PRIVATE.json path)
    'Ding corrected GDA': [('s17', 'out/ding_gda/test_scores_PRIVATE.json')],
    'DAW-FDD': [('s17', 'out/daw/test_scores_PRIVATE.json')],
    'CES-FL P': [('s17', 'out/cesfl_p/test_scores_PRIVATE.json')],
    'L_A (plain-head BCE)': [('s17', 'out/l_a/test_scores_PRIVATE.json')],
    'RF-S (ours)': [('s17', 'out/rf_s_s17/test_scores_PRIVATE.json'), ('s7', 'out/rf_s_s7/test_scores_PRIVATE.json'),
                     ('s27', 'out/rf_s_s27/test_scores_PRIVATE.json')],
    'RF-SJg (ours)': [('s17', 'out/rf_sjg_s17/test_scores_PRIVATE.json'), ('s7', 'out/rf_sjg_s7/test_scores_PRIVATE.json'),
                       ('s27', 'out/rf_sjg_s27/test_scores_PRIVATE.json')],
}
BASE = Path('/home/sachin.chaudhary/ces_fl_runs/ff2_test_eval_20261108')


def clean(v): return str(v).strip() if v is not None else 'unknown'


def key_ff_ids8(r):
    g = clean(r.get('gender')); a = clean(r.get('age_bin')); race = clean(r.get('race'))
    if g not in ('Male', 'Female') or a not in ('<30', '30-44', '45-59', '60+') or race not in ('White', 'Asian', 'Black', 'Others'):
        return None
    return f"{g}|{'Youth' if a == '<30' else 'Adult'}|{'White' if race == 'White' else 'NonWhite'}"


def key_age_skin(r):
    a = clean(r.get('age_bin')); s = clean(r.get('skin_bin'))
    if a not in ('<30', '30-44', '45-59', '60+') or s not in ('0', '1', '2', '3'): return None
    return f'{a}|skin{s}'


GROUPS = {
    'gender': lambda r: clean(r.get('gender')) if clean(r.get('gender')) in ('Female', 'Male') else None,
    'race': lambda r: clean(r.get('race')) if clean(r.get('race')) in ('Asian', 'Black', 'Others', 'White') else None,
    'gender_race': lambda r: clean(r.get('gender_race')) if clean(r.get('gender_race')) != 'unknown' else None,
    'age_bin': lambda r: clean(r.get('age_bin')) if clean(r.get('age_bin')) in ('<30', '30-44', '45-59', '60+') else None,
    'skin_bin_known': lambda r: clean(r.get('skin_bin')) if clean(r.get('skin_bin')) in ('0', '1', '2', '3') else None,
    'age_x_skin16': key_age_skin,
    'sex_age_race8_all': key_ff_ids8,
    'sex_age_race8_min20': key_ff_ids8,
}


def compute(rows, z, t, group_fn, min_support):
    y = np.asarray([int(r['label']) for r in rows]); z = np.asarray(z, dtype=float); pred = z >= t
    if z.shape != y.shape or not np.isfinite(z).all(): raise ValueError('Bad score alignment')
    real = y == 0; fake = y == 1
    overall = {'n': len(y), 'n_real': int(real.sum()), 'n_fake': int(fake.sum()),
               'auc': float(roc_auc_score(y, z)), 'fpr': float(pred[real].mean()), 'fnr': float((~pred[fake]).mean()),
               'tpr': float(pred[fake].mean()), 'accuracy': float((pred == y).mean()), 'positive_rate': float(pred.mean())}
    buckets = {}
    for r, yy, zz, pp in zip(rows, y, z, pred):
        g = group_fn(r)
        if g is None: continue
        b = buckets.setdefault(g, {'y': [], 'z': [], 'p': []}); b['y'].append(yy); b['z'].append(zz); b['p'].append(pp)
    stats = {}
    for g, b in buckets.items():
        yy = np.asarray(b['y']); zz = np.asarray(b['z']); pp = np.asarray(b['p']); r = yy == 0; f = yy == 1
        nreal, nfake = int(r.sum()), int(f.sum())
        d = {'n': len(yy), 'n_real': nreal, 'n_fake': nfake,
             'auc': float(roc_auc_score(yy, zz)) if r.any() and f.any() else None,
             'fpr': float(pp[r].mean()) if r.any() else None,
             'tpr': float(pp[f].mean()) if f.any() else None,
             'fnr': float((~pp[f]).mean()) if f.any() else None,
             'accuracy': float((pp == yy).mean()) if len(yy) else None,
             'positive_rate': float(pp.mean()) if len(yy) else None}
        stats[g] = d
    eligible = {g: d for g, d in stats.items() if d['n_real'] >= min_support and d['n_fake'] >= min_support}
    if len(eligible) < 2: return overall, stats, eligible, {}
    fprs = [d['fpr'] for d in eligible.values()]; tprs = [d['tpr'] for d in eligible.values()]
    prs = [d['positive_rate'] for d in eligible.values()]; acc = [d['accuracy'] for d in eligible.values()]
    fpr_gap = max(fprs) - min(fprs); tpr_gap = max(tprs) - min(tprs)
    fair = {'n_groups': len(eligible),
            'F_FPR': sum(abs(d['fpr'] - overall['fpr']) for d in eligible.values()),
            'F_DP': max(prs) - min(prs),
            'es_AUC': overall['auc'] / (1 + sum(abs(d['auc'] - overall['auc']) for d in eligible.values())),
            'FPR_gap': fpr_gap, 'TPR_gap': tpr_gap, 'F_MEO_max': max(fpr_gap, tpr_gap), 'F_OAE': max(acc) - min(acc),
            'worst_group_AUC': min(d['auc'] for d in eligible.values())}
    return overall, stats, eligible, fair


metric_rows = []
support_rows = []
for model, seeds in SPECS.items():
    for seed, rel in seeds:
        d = json.load(open(BASE / rel))
        if d['manifest_sha256'] != TEST_SHA or d['split'] != 'test':
            raise ValueError(f'{model} {seed}: not bound to the canonical TEST manifest')
        rows = d['rows']; z = d['logits']
        if len(rows) != N_TEST or len(z) != N_TEST:
            raise ValueError(f'{model} {seed}: TEST row count mismatch ({len(rows)} / {len(z)}, expected {N_TEST})')
        thr = float(d.get('threshold', d.get('threshold_logit')))
        mini = int(d.get('min_support', MIN_SUPPORT))
        overall_for_view = None
        for axis, fn in GROUPS.items():
            mins = 1 if axis == 'sex_age_race8_all' else mini
            overall, stats, eligible, fair = compute(rows, z, thr, fn, mins)
            if overall_for_view is None: overall_for_view = overall
            metric_rows.append({'model': model, 'seed': seed, 'threshold_view': 'TEST_frozen', 'axis': axis,
                'eligible_groups': fair.get('n_groups', 0), 'min_support_per_class': mins,
                'overall_auc_pct': 100 * overall['auc'], 'overall_fpr_pct': 100 * overall['fpr'], 'overall_fnr_pct': 100 * overall['fnr'],
                'F_FPR_pct': 100 * fair['F_FPR'], 'F_DP_pct': 100 * fair['F_DP'], 'es_AUC_pct': 100 * fair['es_AUC'],
                'FPR_gap_pp': 100 * fair['FPR_gap'], 'TPR_gap_pp': 100 * fair['TPR_gap'], 'F_MEO_max_pp': 100 * fair['F_MEO_max'],
                'F_OAE_pp': 100 * fair['F_OAE'], 'worst_group_auc_pct': 100 * fair['worst_group_AUC']})
            for g, s in stats.items():
                support_rows.append({'model': model, 'seed': seed, 'threshold_view': 'TEST_frozen', 'axis': axis, 'group': g,
                    'n': s['n'], 'n_real': s['n_real'], 'n_fake': s['n_fake'], 'eligible': g in eligible,
                    'auc_pct': 100 * s['auc'] if s['auc'] is not None else '', 'fpr_pct': 100 * s['fpr'] if s['fpr'] is not None else '',
                    'tpr_pct': 100 * s['tpr'] if s['tpr'] is not None else '', 'fnr_pct': 100 * s['fnr'] if s['fnr'] is not None else '',
                    'accuracy_pct': 100 * s['accuracy'] if s['accuracy'] is not None else '',
                    'positive_rate_pct': 100 * s['positive_rate'] if s['positive_rate'] is not None else ''})

for name, rows_out in [('METRICS_LONG_TEST.csv', metric_rows), ('GROUP_SUPPORT_TEST.csv', support_rows)]:
    p = OUT / name
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)

print(json.dumps({'status': 'DONE', 'models': list(SPECS), 'metric_rows': len(metric_rows), 'support_rows': len(support_rows),
                  'test_manifest_sha256': TEST_SHA, 'n_test_rows': N_TEST}))
