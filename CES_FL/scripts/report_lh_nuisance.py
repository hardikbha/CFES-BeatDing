#!/usr/bin/env python3
"""Compare a complete four-arm nuisance pilot from native validation receipts."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]/'src/ces_fl_nuisance_resnet50.py'
spec = importlib.util.spec_from_file_location('nuisance_model', SRC)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ARMS = {'A0': ('A','none','legacy'), 'H0': ('D','none','hierarchical_real'),
        'A1': ('A','matched','legacy'), 'H1': ('D','matched','hierarchical_real')}


def report(runs, out):
    if set(runs) != set(ARMS): raise ValueError('Require all four arms; report partial progress separately')
    payload = {}; common = None
    for name, (arm, nuisance, error) in ARMS.items():
        root = Path(runs[name])
        if not (root/'DONE').is_file(): raise ValueError('Run not complete: '+name)
        v = json.loads((root/'RUN_VERIFIED.json').read_text())
        c = json.loads((root/'config.json').read_text())
        h = json.loads((root/'history.json').read_text())
        scores = json.loads((root/'validation_scores_PRIVATE.json').read_text())
        if v.get('status') != 'COMPLETED_VERIFIED' or v.get('synthetic_fixture') or v.get('test_evaluated'):
            raise ValueError('Invalid production verification: '+name)
        if v['source_sha256'] != m.sha(SRC) or v['config_digest'] != m.digest(c): raise ValueError('Source/config mismatch')
        if (v['arm'],c['nuisance_mode'],c['error_control']) != (arm,nuisance,error): raise ValueError('Arm mismatch')
        if c['epochs'] != 3 or v['epochs_completed'] != 3 or [r['epoch'] for r in h] != [1,2,3]:
            raise ValueError('Incomplete or altered epoch budget')
        winner = max(h, key=lambda r:r['validation']['overall']['auc'])
        if winner['epoch'] != v['selected_epoch'] or winner['validation'] != v['validation']:
            raise ValueError('Selection differs from native earliest maximum AUC')
        if m.sha(root/'best.pt') != v['selected_checkpoint_sha256'] or scores['checkpoint_sha256'] != v['selected_checkpoint_sha256']:
            raise ValueError('Checkpoint binding changed')
        if scores['split'] != 'validation' or scores['test_read'] or scores['manifest_sha256'] != c['val_sha256']:
            raise ValueError('Not the frozen validation cache')
        replay = m.metric_report(scores['rows'],scores['logits'],scores['threshold'],c['min_eval_support'])
        if replay != v['validation']: raise ValueError('Saved score replay differs')
        config_common = {k:x for k,x in c.items() if k not in ('nuisance_mode','error_control')}
        if common is None: common = config_common
        elif config_common != common: raise ValueError('Unmatched scientific configuration')
        if any(not r['originals_once_each'] or sum(r['main_nuisance_counts'].values()) != r['unique_originals'] for r in h):
            raise ValueError('Source/augmentation coverage mismatch')
        fixed = m.metric_report(scores['rows'],scores['logits'],0.,c['min_eval_support'])
        payload[name] = {'verification':v,'history':h,'metrics':m.flat_metrics(v['validation']),
                         'fixed_probability_0_5':fixed}
    # Main image-transform draws must match across objective choices.
    for left,right in [('A0','H0'),('A1','H1')]:
        for a,b in zip(payload[left]['history'],payload[right]['history']):
            if a['main_nuisance_by_axis_group_class'] != b['main_nuisance_by_axis_group_class']:
                raise ValueError('Original nuisance/class/group draws differ')
    first = json.loads((Path(runs['A0'])/'validation_scores_PRIVATE.json').read_text())
    for name in ('H0','A1','H1'):
        other = json.loads((Path(runs[name])/'validation_scores_PRIVATE.json').read_text())
        if first['rows'] != other['rows']: raise ValueError('Validation row membership/order differs')
    keys = payload['A0']['metrics'].keys(); differences = {}
    for label,left,right in [('H0_minus_A0','H0','A0'),('H1_minus_A1','H1','A1'),('A1_minus_A0','A1','A0'),('H1_minus_H0','H1','H0')]:
        differences[label] = {k:None if payload[left]['metrics'][k] is None or payload[right]['metrics'][k] is None
                             else payload[left]['metrics'][k]-payload[right]['metrics'][k] for k in keys}
    interaction = {k: None if differences['H1_minus_A1'][k] is None or differences['H0_minus_A0'][k] is None
                   else differences['H1_minus_A1'][k]-differences['H0_minus_A0'][k] for k in keys}
    out = m.fresh_dir(out)
    m.write_json(out/'COMPARISON.json', {'arms':payload,'contrasts':differences,'interaction':interaction,
        'interaction_definition':'(H1-A1)-(H0-A0), point estimate only',
        'test_read':False,'new_model_inference':False,'source_sha256':m.sha(SRC),
        'interpretation':'Clean source-validation only; no proven cross-domain gain or disentanglement. Single seed.'})
    lines=['# Four-arm nuisance pilot — FF++ validation only','',
           '| Arm | Selected epoch | AUC % | FPR % | FNR % | Gender F_FPR | Race F_FPR | Joint F_FPR | Age F_FPR | Skin F_FPR |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,d in payload.items():
        v=d['verification'];r=v['validation'];o=r['overall'];a=r['axes']
        values=[f'{100*o[k]:.6f}' for k in ('auc','fpr','fnr')]+[f'{a[k]["F_FPR"]:.6f}' for k in ('gender','race','gender_race','age_bin','skin_bin')]
        lines.append('|'+name+'|'+str(v['selected_epoch'])+'|'+'|'.join(values)+'|')
    lines += ['', 'Fairness values are fractions; no per-metric checkpoint selection. All contrasts and histories are in COMPARISON.json.',
              'Separate fixed-probability0.5 point diagnostics, including every group, are also in COMPARISON.json. They do not replace the primary stored validation thresholds.',
              'Use native paired connected-contributor bootstrap for H1−A1 and H1−H0, keep the 900/1000 usable-draw requirement, and report sparse cells as unsupported.',
              'The four-arm interaction has no confidence interval in this helper. A negative interaction for disparity is descriptive, not a statistical or causal conclusion.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('FOUR_ARM_VALIDATION_COMPARISON_COMPLETE',out)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs-json',type=Path,required=True);p.add_argument('--out',required=True)
    a=p.parse_args();report(json.loads(a.runs_json.read_text()),a.out)
