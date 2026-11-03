#!/usr/bin/env python3
"""CPU-only aggregate VAL reporting from verified selected checkpoints; no inference."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('ff2',ROOT/'CES_FL/src/ff2_fairness_baselines.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def load(path):
    report=json.loads((path/'RUN_VERIFIED.json').read_text())
    scores=json.loads((path/'validation_scores_PRIVATE.json').read_text())
    c=json.loads((path/'config.json').read_text())
    if not (path/'DONE').exists() or report['status']!='COMPLETED_VERIFIED' or report['fixture'] or report['source_hashes']!=m.hashes():
        raise ValueError('Unverified, synthetic or stale run')
    if scores['split']!='validation' or scores['manifest_sha256']!=c['val_sha256']:raise ValueError('Wrong split')
    if m.common.sha(path/'best.pt')!=report['checkpoint_sha256'] or scores['checkpoint_sha256']!=report['checkpoint_sha256']:
        raise ValueError('Changed selected checkpoint')
    actual=m.common.metric_report(scores['rows'],scores['logits'],scores['threshold'],scores['min_support'])
    if actual!=report['validation']:raise ValueError('Private predictions do not replay published validation')
    return report,scores


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--bootstrap',type=int,default=1000);a=p.parse_args()
    if a.bootstrap<1000:p.error('Use at least1000 predeclared resamples for the benchmark report')
    out=m.common.fresh_dir(a.out);runs={};missing=[]
    for method in m.METHODS:
        path=Path(a.work_root)/method/'training'
        if not (path/'DONE').is_file():missing.append(method);continue
        runs[method]=load(path)
    aggregate={'split':'validation','test_read':False,'source_hashes':m.hashes(),'missing_or_incomplete':missing,
               'comparison':'common-recipe FF2; no paper-test claim','methods':{}}
    for method,(report,scores) in runs.items():
        item={'verified':report,'uncertainty':m.common.bootstrap(scores['rows'],scores['logits'],scores['threshold'],scores['min_support'],a.bootstrap,17)}
        if method!='bce' and 'bce' in runs:
            baseline=runs['bce'][1]
            if scores['rows']!=baseline['rows'] or scores['manifest_sha256']!=baseline['manifest_sha256']:raise ValueError('Unpaired validation rows')
            item['paired_difference_vs_bce']=m.common.bootstrap(scores['rows'],scores['logits'],scores['threshold'],scores['min_support'],a.bootstrap,17,
                other=(baseline['logits'],baseline['threshold']))
        aggregate['methods'][method]=item
    m.common.write_json(out/'VALIDATION_AGGREGATES.json',aggregate)
    lines=['# FairForge-2 common-recipe validation comparison','',
        'All cells below use the same selected checkpoint per method. Values are percentages (fraction ×100). No TEST results. Lower F_FPR/F_DP is better; higher AUC/es_AUC is better.','',
        '| Method | Axis | Overall AUC | F_FPR | F_DP | es_AUC | Eligible groups |',
        '|---|---|---:|---:|---:|---:|---:|']
    def cell(v):return 'unavailable' if v is None else f'{100*v:.3f}'
    for method,(r,_) in runs.items():
        for axis,d in r['validation']['axes'].items():
            lines.append(f"| {method} | {axis} | {cell(r['validation']['overall']['auc'])} | {cell(d['F_FPR'])} | {cell(d['F_DP'])} | {cell(d['es_AUC'])} | {len(d['eligible_groups'])} |")
    lines+=['','Missing/incomplete: '+(', '.join(missing) or 'none')+'.',
        '', 'JSON includes subgroup counts, fixed-logit0 metrics, connected-contributor bootstrap intervals and paired differences versus BCE. Thresholds and subgroup eligibility remain fixed during resampling. Invalid replicate counts must be reported; these intervals do not measure training-seed uncertainty. Gender × race means joint groups, never multiplication of marginal FPRs.',
        '', 'The existing Ding FF2 run uses a different adapter, sampler and two-stage schedule. It is a separately labeled comparison, not the matched native BCE control.']
    (out/'VALIDATION_TABLE.md').write_text('\n'.join(lines)+'\n')
    print(str(out))


if __name__=='__main__':main()
