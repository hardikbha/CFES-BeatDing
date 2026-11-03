#!/usr/bin/env python3
"""Render measured score reports only; separate validation/TEST and thresholds."""
import argparse
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',nargs='+',type=Path,required=True);p.add_argument('--profiles',nargs='*',type=Path,default=[])
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    cards={r['id']:r for r in json.loads((HERE/'methods.json').read_text())['methods']}
    profiles={json.loads(p.read_text())['method']:json.loads(p.read_text()) for p in a.profiles}
    records=[];seen=set()
    for path in a.reports:
        r=json.loads(path.read_text());method=r['method'];key=(r['split'],method)
        if r.get('synthetic_fixture') is not False:raise ValueError('Only explicitly verified non-fixture score reports belong in benchmark tables')
        if key in seen:raise ValueError('One selected checkpoint per method/split; no best-per-cell merging')
        seen.add(key)
        card=cards['xception' if method=='bce' else method]
        profile=profiles.get('xception' if method in ('bce','dag','daw') else method,{})
        for view,v in r['views'].items():
            row={'split':r['split'],'view':view,'category':card['category'],'method':card['name'],
                 'checkpoint':r['checkpoint_sha256'],'Params_M':None if not profile else profile['deployed_total_parameters']/1e6,
                 'Trainable_M':None if not profile else profile['trainable_parameters']/1e6,
                 'GFLOPs_counted':profile.get('counted_gflops'),
                 'FLOPs_coverage':'COMPLETE' if profile.get('complete_flop_coverage') else 'PARTIAL' if profile else 'PENDING'}
            row.update({k:None if v['overall'][k] is None else 100*v['overall'][k] for k in ('auc','accuracy','ap','eer','fpr','fnr')})
            for axis,av in v['axes'].items():
                for metric in ('F_FPR','F_DP','F_MEO','F_EO','F_OAE','es_AUC'):
                    row[axis+'/'+metric]=None if av[metric] is None else 100*av[metric]
            records.append(row)
    columns=list(records[0])
    with (a.out/'ALL_METRICS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(records)
    for split,view in sorted({(r['split'],r['view']) for r in records}):
        subset=[r for r in records if r['split']==split and r['view']==view]
        subset.sort(key=lambda r:({'CNN':0,'Transformer':1,'VLM':2,'Fairness':3}.get(r['category'],4),r['method']))
        cols=['category','method','auc','fpr','fnr','gender_race/F_FPR','gender_race/F_MEO','gender_race/F_DP','gender_race/F_OAE','Params_M','Trainable_M','GFLOPs_counted','FLOPs_coverage']
        text=[f'# FairForge-2 {split}: {view}','',
              'Measured selected-model results only. Fairness/utility values ×100; parameter counts in millions. FLOPs use fvcore one-FMA convention at224pixels; PARTIAL means unsupported operations remain. Missing values are unavailable, not zero. Native architecture/recipe adaptations are in the method cards.','',
              '| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
        for row in subset:
            text.append('| '+' | '.join('—' if row[k] is None else f'{row[k]:.4f}' if isinstance(row[k],float) else str(row[k]) for k in cols)+' |')
        (a.out/f'{split}_{view}.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
