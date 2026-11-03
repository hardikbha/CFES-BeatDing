#!/usr/bin/env python3
"""Common utility/fairness reporting from fixed saved scores; never fits a threshold."""
import argparse
import importlib.util
import json
import hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

AXES = ('gender', 'race', 'gender_race', 'age_bin', 'skin_bin', 'ids8', 'gender_skin')


def utility(y, z, t):
    pred = z >= t; real = y == 0; fake = y == 1
    auc = ap = eer = None
    if real.any() and fake.any():
        auc = float(roc_auc_score(y, z)); ap = float(average_precision_score(y, z))
        fpr, tpr, _ = roc_curve(y, z, drop_intermediate=False); fnr = 1 - tpr
        d = fpr - fnr; i = int(np.flatnonzero(d >= 0)[0])
        if d[i] == 0 or i == 0:
            eer = float(fpr[i])
        else:
            alpha = -d[i-1] / (d[i] - d[i-1])
            eer = float(fpr[i-1] + alpha * (fpr[i] - fpr[i-1]))
    return {'n': int(len(y)), 'n_real': int(real.sum()), 'n_fake': int(fake.sum()),
            'auc': auc, 'ap': ap, 'eer': eer,
            'accuracy': float((pred == y).mean()) if len(y) else None,
            'fpr': float(pred[real].mean()) if real.any() else None,
            'fnr': float((~pred[fake]).mean()) if fake.any() else None,
            'positive_rate': float(pred.mean()) if len(y) else None}


def report(rows, logits, threshold, min_support=20):
    z = np.asarray(logits, dtype=float); y = np.array([r['label'] for r in rows], dtype=int)
    if z.shape != y.shape or not len(y) or not np.isfinite(z).all() or not np.isin(y, [0, 1]).all() or not np.isfinite(threshold):
        raise ValueError('Invalid fixed predictions/labels/threshold')
    overall = utility(y, z, threshold); axes = {}
    for axis in AXES:
        labels = []
        for row in rows:
            g = str(row.get(axis) or 'unknown')
            if axis == 'gender_skin':
                a, b = str(row.get('gender') or 'unknown'), str(row.get('skin_bin') or 'unknown')
                g = a + '|' + b if 'unknown' not in (a, b) else 'unknown'
            labels.append(g)
        labels = np.array(labels)
        groups = {g: utility(y[labels == g], z[labels == g], threshold) for g in np.unique(labels)}
        names = [g for g, v in groups.items() if g != 'unknown' and min(v['n_real'], v['n_fake']) >= min_support]
        good = [groups[g] for g in names]
        v = {'groups': groups, 'eligible_groups': names, 'min_support_per_class': min_support,
             'worst_group_auc': min(g['auc'] for g in good) if good else None}
        for name in ('F_FPR', 'F_DP', 'F_MEO', 'F_EO', 'F_OAE', 'es_AUC'):
            v[name] = None
        if len(good) > 1:
            gap = lambda k: max(g[k] for g in good) - min(g[k] for g in good)
            v.update(F_FPR=sum(abs(g['fpr']-overall['fpr']) for g in good),
                     F_DP=gap('positive_rate'), F_MEO=max(gap('fpr'), gap('fnr')),
                     F_EO=sum(abs(g['fpr']-overall['fpr'])+abs(g['fnr']-overall['fnr']) for g in good),
                     F_OAE=gap('accuracy'),
                     es_AUC=overall['auc']/(1+sum(abs(g['auc']-overall['auc']) for g in good)))
        axes[axis] = v
    return {'units': 'fractions', 'threshold_logit': float(threshold), 'overall': overall, 'axes': axes,
            'eer_is_descriptive_not_a_deployment_threshold': True,
            'gender_skin_uses_existing_bins_not_inferred_AI_Face_six_groups': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scores', type=Path, required=True); p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--method', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=0)
    a = p.parse_args()
    if a.out.exists(): raise ValueError('Preserve previous report')
    s = json.loads(a.scores.read_text()); v = json.loads(a.verification.read_text())
    if v.get('fixture') or v.get('synthetic_fixture'):
        raise ValueError('Synthetic checks cannot become benchmark table rows')
    if s.get('split') not in ('validation', 'test'): raise ValueError('Explicit score split required')
    if v.get('status') not in ('COMPLETED_VERIFIED', 'EVALUATED_FROZEN') or v['checkpoint_sha256'] != s['checkpoint_sha256']:
        raise ValueError('Unverified or mismatched checkpoint')
    if v.get('method') != a.method: raise ValueError('Method does not match native verification receipt')
    result = {'method': a.method, 'split': s['split'], 'checkpoint_sha256': s['checkpoint_sha256'],
              'manifest_sha256': s['manifest_sha256'], 'threshold_search': False,
              'synthetic_fixture': False,
              'scoring_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'views': {'fixed_probability_0_5': report(s['rows'], s['logits'], 0., s['min_support']),
                        'source_validation_10pct': report(s['rows'], s['logits'], s['threshold'], s['min_support'])}}
    if a.bootstrap:
        if a.bootstrap != 1000: raise ValueError('Preserve the declared1000-draw protocol')
        path = Path(__file__).resolve().parents[1] / 'src/ces_fl_head_ablation.py'
        spec = importlib.util.spec_from_file_location('metric_reference', path)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        result['ding_metric_uncertainty'] = {name: m.bootstrap(s['rows'], s['logits'], threshold, s['min_support'], 1000, 17)
                                            for name, threshold in [('fixed_probability_0_5', 0.), ('source_validation_10pct', s['threshold'])]}
        result['uncertainty_scope'] = 'Ding metrics and native axes only; extended EO/OAE/utility metrics are point estimates. At least900usable draws required; one trained seed.'
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
