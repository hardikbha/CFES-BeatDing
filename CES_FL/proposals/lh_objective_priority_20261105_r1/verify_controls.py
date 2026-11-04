#!/usr/bin/env python3
"""Replay existing three-epoch controls from private saved predictions, on CPU."""
import argparse,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('cohort',HERE.parents[1]/'src/ces_fl_nuisance_resnet50.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def verify(root,name,native,expected):
    root=Path(root);v=json.loads((root/'RUN_VERIFIED.json').read_text());c=json.loads((root/'config.json').read_text())
    scores=json.loads((root/'validation_scores_PRIVATE.json').read_text())
    history=json.loads((root/'history.json').read_text())
    if not (root/'DONE').exists() or v['status']!='COMPLETED_VERIFIED' or v['epochs_completed']!=3 or v['arm']!=native or v.get('synthetic_fixture'):
        raise ValueError('Not the requested completed real-data control: '+name)
    if c!=expected or v['source_sha256']!=m.sha(m.__file__):raise ValueError('Control configuration/source mismatch')
    if v['selected_checkpoint_sha256']!=m.sha(root/'best.pt') or scores['checkpoint_sha256']!=v['selected_checkpoint_sha256']:
        raise ValueError('Control checkpoint changed')
    if [h['epoch'] for h in history]!=[1,2,3] or max(history,key=lambda h:h['validation']['overall']['auc'])['epoch']!=v['selected_epoch']:
        raise ValueError('Control selection mismatch')
    if scores['manifest_sha256']!=c['val_sha256'] or len(scores['rows'])!=25386 or scores['threshold']!=v['validation']['threshold_logit']:
        raise ValueError('Control membership/threshold mismatch')
    if m.metric_report(scores['rows'],scores['logits'],scores['threshold'],scores['min_support'])!=v['validation']:
        raise ValueError('Control saved-score replay mismatch')
    return {'arm':name,'checkpoint_sha256':v['selected_checkpoint_sha256'],'selected_epoch':v['selected_epoch'],'validation':v['validation']},scores

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if Path(a.out).exists():raise ValueError('Preserve prior receipt')
    results=[];previous=None
    for name,native in [('A0','A'),('H0','D')]:
        result,scores=verify(Path(a.root)/name,name,native,json.loads((HERE/'evidence'/f'{name}_reference_config.json').read_text()))
        if previous is not None and scores['rows']!=previous:raise ValueError('Control row order/annotations differ')
        previous=scores['rows'];results.append(result)
    m.write_json(a.out,{'status':'PASS','controls':results,'model_inference':False,'test_read':False})
