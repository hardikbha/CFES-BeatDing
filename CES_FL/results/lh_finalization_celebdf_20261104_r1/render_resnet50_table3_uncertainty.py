#!/usr/bin/env python3
"""Render aggregate connected-contributor uncertainty; no row-level scores."""
import argparse
import json
from pathlib import Path

DATASETS=('ffpp','dfdc','celebdf','dfd')
ARMS=('L_A','L_H')
VIEWS=(('selected_FFPP_validation_real_empirical_10pct','Source-VAL10'),('fixed_probability_0.5','Fixed 0.5'))
AXIS_METRICS=('F_FPR','F_DP','es_AUC','worst_group_auc')
METRICS=('overall/auc','overall/fpr','overall/fnr',
         *(f'{axis}/{metric}' for axis in ('gender','race','age_bin','skin_bin','gender_race')
           for metric in AXIS_METRICS))


def pct(x):return 'NA' if x is None else f'{100*float(x):.3f}'


def point_from(metrics,key):
    axis,metric=key.split('/',1)
    if axis=='overall':return metrics['overall'].get(metric)
    return metrics['axes'].get(axis,{}).get(metric)


def interval(ci_obj,floor):
    n=ci_obj.get('valid_replicates',0);ci=ci_obj.get('ci95')
    if ci is None or n<floor:return f'Unavailable (<{floor})' if n else 'Unavailable (no eligible support)'
    return f'[{pct(ci[0])}, {pct(ci[1])}]'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--analysis-root',required=True)
    p.add_argument('--minimum-valid',type=int,default=900)
    p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.analysis_root);out=Path(a.out)
    if out.exists():raise ValueError(f'Preserve existing uncertainty report: {out}')
    lines=['# Table 3 uncertainty — ResNet-50 L_A/L_H, seed 17','',
      'Intervals are percentile connected-recorded-contributor bootstrap intervals for fixed selected checkpoints and frozen thresholds. They are conditional on these single-seed models, not training-seed variability or verified biological-person independence. Fairness intervals are reportable only with at least the prespecified900/1000 valid replicates; lower counts are shown as unavailable, without relaxing the floor.','',
      'All estimates and intervals below are percentages; paired differences are percentage-point differences, L_H minus L_A. AUC and es-AUC are multiplied by100; rates and F_FPR/F_DP retain their metric definitions.','',
      '## Per-arm estimates and intervals','',
      '| Dataset | Arm | Threshold | Metric | Estimate | 95% CI | Valid / 1000 |','|---|---|---|---|---:|---|---:|']
    for ds in DATASETS:
        for arm in ARMS:
            data=json.loads((root/f'{arm}_{ds}.json').read_text())
            for key,view in VIEWS:
                block=data[key];metrics=block['metrics'];intervals=block['uncertainty']['intervals']
                for metric in METRICS:
                    obj=intervals.get(metric,{})
                    estimate=point_from(metrics,metric)
                    lines.append(f"| {ds.upper() if ds!='celebdf' else 'Celeb-DF'} | {arm} | {view} | {metric} | {pct(estimate)} | {interval(obj,a.minimum_valid)} | {obj.get('valid_replicates',0)}/1000 |")
    lines += ['', '## Paired L_H − L_A bootstrap','',
      'The paired bootstrap reuses the same recorded-contributor resampling draws for both fixed models. A positive difference means L_H is higher on that metric; for FPR/FNR/F_FPR lower is preferable, while AUC/es-AUC higher is preferable.','',
      '| Dataset | Threshold | Metric | H−A estimate (pp) | 95% CI (pp) | Valid / 1000 |','|---|---|---|---:|---|---:|']
    paired_metrics=METRICS
    for ds in DATASETS:
        data=json.loads((root/f'H_minus_A_{ds}.json').read_text())
        for key,view in VIEWS:
            result=data[key]
            for metric in paired_metrics:
                obj=result['intervals'].get(metric,{})
                lines.append(f"| {ds.upper() if ds!='celebdf' else 'Celeb-DF'} | {view} | {metric} | {pct(obj.get('estimate'))} | {interval(obj,a.minimum_valid)} | {obj.get('valid_replicates',0)}/1000 |")
    lines += ['', 'Axes with no non-unknown group support are explicitly unavailable (for example age and skin tone in some cross-domain manifests). Sparse intersection intervals with fewer than900valid replicates remain unavailable. These test-set intervals describe sampling uncertainty in the fixed manifest under recorded-contributor clustering; they do not authorize selection or tuning.']
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'PASS','out':str(out),'arms':len(ARMS),'datasets':len(DATASETS),
                      'views':len(VIEWS),'minimum_valid_replicates':a.minimum_valid,'row_level_data_written':False}))


if __name__=='__main__':main()
