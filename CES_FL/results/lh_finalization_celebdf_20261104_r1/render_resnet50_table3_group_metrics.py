#!/usr/bin/env python3
"""Export aggregate per-group TEST FPR/FNR/support; never exports row scores."""
import argparse
import hashlib
import csv
import json
from pathlib import Path

DATASETS=('ffpp','dfdc','celebdf','dfd')
ARMS=('L_A','L_H')
AXES=('gender','race','age_bin','skin_bin','gender_race')
VIEWS=(('calibrated','source_validation_real_10pct'),('fixed_probability_0_5','fixed_probability_0_5'))


def pct(value):return '' if value is None else f'{100*float(value):.6f}'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--eval-root',required=True);p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.eval_root);out=Path(a.out);plan=json.loads(Path(a.plan).read_text());plan_sha=sha(a.plan)
    if plan.get('format')!='CESFL_RESNET50_TABLE3_LOCKED_v1' or set(plan.get('arms',{}))!=set(ARMS) or set(plan.get('datasets',{}))!=set(DATASETS):
        raise ValueError('Invalid locked Table3 plan')
    if out.exists():raise ValueError(f'Preserve existing aggregate export: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    fields=['dataset','arm','threshold_view','axis','group','n','n_real','n_fake','eligible_min_support',
            'auc_pct','fpr_pct','fnr_pct','positive_rate_pct']
    with out.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for arm in ARMS:
            for dataset in DATASETS:
                r=root/arm/dataset
                receipt=json.loads((r/'INFERENCE_VERIFIED.json').read_text())
                if (receipt.get('status')!='PASS' or receipt.get('test_calibration') is not False
                        or receipt.get('test_threshold_search') is not False or receipt.get('plan_sha256')!=plan_sha):
                    raise ValueError(f'Invalid locked inference receipt: {arm}/{dataset}')
                metrics=json.loads((r/'metrics.json').read_text())
                if (sha(r/'metrics.json')!=receipt.get('metrics_sha256') or sha(r/'scores_PRIVATE.json')!=receipt.get('scores_sha256')
                        or metrics.get('checkpoint_sha256')!=plan['arms'][arm]['checkpoint_sha256']
                        or metrics.get('manifest_sha256')!=plan['datasets'][dataset]['sha256']):
                    raise ValueError(f'Inference outputs do not match bound plan: {arm}/{dataset}')
                for key,view in VIEWS:
                    report=metrics[key]
                    for axis in AXES:
                        a_report=report['axes'][axis]
                        eligible=set(a_report.get('eligible_groups',[]))
                        for group,g in a_report['groups'].items():
                            w.writerow({'dataset':dataset,'arm':arm,'threshold_view':view,'axis':axis,'group':group,
                                'n':g['n'],'n_real':g['n_real'],'n_fake':g['n_fake'],
                                'eligible_min_support':group in eligible,'auc_pct':pct(g.get('auc')),
                                'fpr_pct':pct(g.get('fpr')),'fnr_pct':pct(g.get('fnr')),
                                'positive_rate_pct':pct(g.get('positive_rate'))})
    print(json.dumps({'status':'PASS','out':str(out),'rows_written':sum(1 for _ in out.open())-1,
                      'contains_image_paths':False,'contains_row_scores':False}))


if __name__=='__main__':main()
