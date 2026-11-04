#!/usr/bin/env python3
"""Render aggregate-only ResNet Table-3 point estimates from locked receipts."""
import argparse
import hashlib
import json
from pathlib import Path

DATASETS=('ffpp','dfdc','celebdf','dfd')
ARMS=('L_A','L_H')
DING_PUBLISHED={
 'ffpp':{'gender':(3.940,5.140,94.460),'race':(18.670,2.430,92.690),'joint':(34.300,8.610,82.870),'auc':95.530},
 'dfdc':{'gender':(3.970,3.610,56.960),'race':(11.940,5.150,50.990),'joint':(32.140,10.270,37.550),'auc':62.540},
 'celebdf':{'gender':(4.530,12.000,68.980),'race':(6.760,8.160,71.030),'joint':(5.560,9.110,68.250),'auc':77.350},
 'dfd':{'gender':(3.120,1.760,77.680),'race':(6.510,3.030,70.930),'joint':(20.990,4.850,66.190),'auc':78.630},
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def pct(x):return f'{100*float(x):.3f}'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,help='Private table3_eval output root')
    p.add_argument('--plan',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.root);plan=json.loads(Path(a.plan).read_text())
    if plan.get('format')!='CESFL_RESNET50_TABLE3_LOCKED_v1' or set(plan.get('arms',{}))!=set(ARMS) or set(plan.get('datasets',{}))!=set(DATASETS):
        raise ValueError('Plan does not bind the locked ResNet two-arm/four-dataset table')
    if plan.get('test_threshold_search') is not False or plan.get('target_domain_calibration') is not False:
        raise ValueError('Forbidden TEST tuning policy in locked plan')
    records={}
    for arm in ARMS:
        for ds in DATASETS:
            out=root/arm/ds
            receipt=json.loads((out/'INFERENCE_VERIFIED.json').read_text())
            if receipt.get('status')!='PASS' or receipt.get('test_calibration') is not False or receipt.get('test_threshold_search') is not False:
                raise ValueError(f'{arm}/{ds}: inference receipt failed or declares tuning')
            if receipt.get('plan_sha256')!=sha(a.plan):raise ValueError(f'{arm}/{ds}: plan hash mismatch')
            if sha(out/'metrics.json')!=receipt['metrics_sha256'] or sha(out/'scores_PRIVATE.json')!=receipt['scores_sha256']:
                raise ValueError(f'{arm}/{ds}: inference artifact hash mismatch')
            m=json.loads((out/'metrics.json').read_text())
            if m['checkpoint_sha256']!=plan['arms'][arm]['checkpoint_sha256'] or m['manifest_sha256']!=plan['datasets'][ds]['sha256']:
                raise ValueError(f'{arm}/{ds}: checkpoint/manifest not bound to locked plan')
            records[(arm,ds)]=m
    dest=Path(a.out)
    if dest.exists():raise ValueError(f'Preserve existing table snapshot: {dest}')
    lines=['# Table 3 — ResNet-50 current L_A/L_H results','',
      f"Locked plan SHA256: `{sha(a.plan)}`. This is an aggregate-only rendering from eight one-pass TEST receipts; no threshold tuning or checkpoint selection used TEST.",'',
      'Ding rows are published ResNet-50 references under the paper protocol. CES-FL rows are the new seed-17 ResNet-50 runs and use their frozen FF++ validation-real 10% FPR threshold on all four datasets. The protocols differ; these rows do not establish a protocol-matched win. Point estimates are shown below; connected-contributor confidence intervals are being calculated separately.','',
      '## Source-validation-real 10% FPR threshold', '',
      '| Dataset | Method | Backbone | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Intersection F_FPR ↓ | Intersection F_DP ↓ | Intersection es-AUC ↑ | AUC ↑ |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        d=DING_PUBLISHED[ds]
        lines.append(f"| {ds.upper() if ds!='celebdf' else 'Celeb-DF'} | Ding et al. (published reference) | ResNet-50 | {d['gender'][0]:.3f} | {d['gender'][1]:.3f} | {d['gender'][2]:.3f} | {d['race'][0]:.3f} | {d['race'][1]:.3f} | {d['race'][2]:.3f} | {d['joint'][0]:.3f} | {d['joint'][1]:.3f} | {d['joint'][2]:.3f} | {d['auc']:.3f} |")
        for arm in ARMS:
            m=records[(arm,ds)]['calibrated'];g=m['axes']['gender'];r=m['axes']['race'];j=m['axes']['gender_race']
            lines.append(f"| {ds.upper() if ds!='celebdf' else 'Celeb-DF'} | CES-FL ResNet {arm} [seed17] | ResNet-50 | {pct(g['F_FPR'])} | {pct(g['F_DP'])} | {pct(g['es_AUC'])} | {pct(r['F_FPR'])} | {pct(r['F_DP'])} | {pct(r['es_AUC'])} | {pct(j['F_FPR'])} | {pct(j['F_DP'])} | {pct(j['es_AUC'])} | {pct(m['overall']['auc'])} |")
    lines += ['', '## Fixed probability 0.5', '',
      '| Dataset | Method | Backbone | Gender F_FPR ↓ | Gender F_DP ↓ | Gender es-AUC ↑ | Race F_FPR ↓ | Race F_DP ↓ | Race es-AUC ↑ | Intersection F_FPR ↓ | Intersection F_DP ↓ | Intersection es-AUC ↑ | AUC ↑ |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        for arm in ARMS:
            m=records[(arm,ds)]['fixed_probability_0_5'];g=m['axes']['gender'];r=m['axes']['race'];j=m['axes']['gender_race']
            lines.append(f"| {ds.upper() if ds!='celebdf' else 'Celeb-DF'} | CES-FL ResNet {arm} [seed17] | ResNet-50 | {pct(g['F_FPR'])} | {pct(g['F_DP'])} | {pct(g['es_AUC'])} | {pct(r['F_FPR'])} | {pct(r['F_DP'])} | {pct(r['es_AUC'])} | {pct(j['F_FPR'])} | {pct(j['F_DP'])} | {pct(j['es_AUC'])} | {pct(m['overall']['auc'])} |")
    lines += ['', '## Overall operating points and support', '',
      '| Dataset | Arm | Threshold view | N | Real N | Fake N | AUC | FPR | FNR | Worst gender×race AUC | Eligible gender×race groups |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        for arm in ARMS:
            for key,view in [('calibrated','Source-VAL10'),('fixed_probability_0_5','Fixed 0.5')]:
                m=records[(arm,ds)][key];o=m['overall'];j=m['axes']['gender_race']
                lines.append(f"| {ds.upper() if ds!='celebdf' else 'Celeb-DF'} | {arm} | {view} | {o['n']} | {o['n_real']} | {o['n_fake']} | {pct(o['auc'])} | {pct(o['fpr'])} | {pct(o['fnr'])} | {pct(j['worst_group_auc'])} | {len(j['eligible_groups'])} |")
    axes=(('gender','Gender'),('race','Race'),('age_bin','Age'),('skin_bin','Skin tone'),('gender_race','Gender×race'))
    lines += ['', '## All demographic axes (point estimates)', '',
      'F_FPR/F_DP/es-AUC are unavailable where the test manifest has no defensible labels or eligible groups. Group eligibility uses the evaluator minimum support of20 per class.', '',
      '| Dataset | Arm | Threshold | Axis | F_FPR ↓ | F_DP ↓ | es-AUC ↑ | Worst-group AUC ↑ | Eligible groups |',
      '|---|---|---|---|---:|---:|---:|---:|---|']
    for ds in DATASETS:
        for arm in ARMS:
            for key,view in [('calibrated','Source-VAL10'),('fixed_probability_0_5','Fixed 0.5')]:
                m=records[(arm,ds)][key]
                for axis,label in axes:
                    a=m['axes'][axis]; eligible=', '.join(a['eligible_groups']) or 'none (labels/support unavailable)'
                    def value(name):
                        v=a.get(name)
                        return 'NA' if v is None else pct(v)
                    lines.append(f"| {ds.upper() if ds!='celebdf' else 'Celeb-DF'} | {arm} | {view} | {label} | {value('F_FPR')} | {value('F_DP')} | {value('es_AUC')} | {value('worst_group_auc')} | {eligible} |")
    lines += ['', '## Per-group support', '',
      'Support counts are shared across L_A/L_H because both use the same held-out manifests. `Eligible` indicates both classes meet the20-per-class minimum for group AUC/fairness summaries.', '',
      '| Dataset | Axis | Group | N | Real N | Fake N | Eligible |',
      '|---|---|---|---:|---:|---:|---|']
    for ds in DATASETS:
        ref=records[('L_A',ds)]['calibrated']
        for axis,label in axes:
            a=ref['axes'][axis]; eligible=set(a['eligible_groups'])
            for group,g in a['groups'].items():
                lines.append(f"| {ds.upper() if ds!='celebdf' else 'Celeb-DF'} | {label} | {group} | {g['n']} | {g['n_real']} | {g['n_fake']} | {'yes' if group in eligible else 'no'} |")
    lines += ['', '## Provenance', '',
      f"- L_A checkpoint: `{plan['arms']['L_A']['checkpoint_sha256']}` (selected epoch {plan['arms']['L_A']['selected_epoch']}).",
      f"- L_H checkpoint: `{plan['arms']['L_H']['checkpoint_sha256']}` (selected epoch {plan['arms']['L_H']['selected_epoch']}).",
      '- The TEST scores are held privately on the cluster; only aggregate metrics are rendered here.',
      '- Confidence intervals are not in this point-estimate snapshot. Use the separate paired connected-recorded-contributor bootstrap analysis; report its valid-replicate counts and retain the prespecified 900/1000 interval floor.',
      '- AUC is threshold-independent. FPR/FNR and fairness rows are measured separately at each frozen threshold.',
      '- No direct Ding model was run for this Table 3 extension; Ding values are published references.']
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'PASS','out':str(dest),'plan_sha256':sha(a.plan),'rows':len(ARMS)*len(DATASETS)*2,
                      'test_inferences_verified':len(records),'test_selection':False,'test_calibration':False}))


if __name__=='__main__':main()
