#!/usr/bin/env python3
"""Extended fairness from verified L_H/L_A saved scores; no new inference."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
core=load(ROOT/'CES_FL/src/ces_fl_head_ablation.py','lh_report_core')
metrics=load(ROOT/'CES_FL/scripts/ff2_expanded_metrics.py','lh_extended_metrics')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--scores',type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,choices=[0,1000],default=1000);a=p.parse_args()
    if a.out.exists():raise ValueError('Preserve earlier reports')
    request=json.loads((HERE/'request.json').read_text())
    v=json.loads((a.run/'RUN_VERIFIED.json').read_text());s=json.loads(a.scores.read_text())
    if not (a.run/'DONE').exists() or v['status']!='COMPLETED_VERIFIED' or v['epochs_completed']!=10 or v.get('synthetic_fixture'):
        raise ValueError('Complete verified real-data L_H/L_A run required')
    if v['source_sha256']!=request['source_sha256'] or v['selected_checkpoint_sha256']!=core.sha(a.run/'best.pt') or s['checkpoint_sha256']!=v['selected_checkpoint_sha256']:
        raise ValueError('Score/source/checkpoint mismatch')
    if s['manifest_sha256']!=request['splits'][a.split]['sha256']:raise ValueError('Wrong score split')
    if len(s['rows'])!=request['splits'][a.split]['rows'] or s['threshold']!=v['validation']['threshold_logit']:
        raise ValueError('Score membership or selected validation threshold changed')
    if v['arm'] not in ('A','D'):raise ValueError('Wrong objective arm')
    result={'method':'L_H' if v['arm']=='D' else 'L_A','split':'validation' if a.split=='val' else 'test',
            'checkpoint_sha256':v['selected_checkpoint_sha256'],'source_score_sha256':core.sha(a.scores),
            'selected_epoch':v['selected_epoch'],'synthetic_fixture':False,'threshold_search':False,
            'metric_note':'F_MEO here is MAX equalized-odds gap; exact FairForensics mean convention not assumed.',
            'views':{name:metrics.report(s['rows'],s['logits'],threshold,s['min_support']) for name,threshold in
                     [('fixed_probability_0_5',0.),('source_validation_10pct',s['threshold'])]}}
    if a.bootstrap:
        result['native_ding_metric_intervals']={name:core.bootstrap(s['rows'],s['logits'],threshold,s['min_support'],1000,17)
             for name,threshold in [('fixed_probability_0_5',0.),('source_validation_10pct',s['threshold'])]}
        result['uncertainty_scope']='Native Ding metrics/axes only; extended metrics point estimates; require900usable draws; one trained seed.'
    core.write_json(a.out,result)
