#!/usr/bin/env python3
"""One frozen-checkpoint FF2 TEST evaluation after native completion/replay."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]/'src'
spec=importlib.util.spec_from_file_location('expanded_eval',SRC/'ff2_expanded_runner.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def evaluate(run,preflight,sources,out,device):
    if device.type=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError('Exactly one bound CUDA device required')
    run=Path(run); checkpoint=run/'best.pt'
    if not (run/'DONE').is_file(): raise ValueError('Completed native run required')
    verified=json.loads((run/'RUN_VERIFIED.json').read_text())
    before=r.common.sha(checkpoint)
    if verified.get('status')!='COMPLETED_VERIFIED' or verified.get('fixture') or verified['checkpoint_sha256']!=before:
        raise ValueError('Checkpoint completion/replay/fixture mismatch')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True);c=ck['config']
    if ck.get('fixture'): raise ValueError('No synthetic benchmark scores')
    if verified['epochs']!=50 or ck['selected_epoch']!=verified['selected_epoch'] or c!=json.loads((run/'config.json').read_text()):
        raise ValueError('Incomplete/mismatched selected model')
    legacy=ck['format']=='FF2_COMMON_RECIPE_FAIRNESS_v1'
    if not legacy and ck['format']!=r.VERSION: raise ValueError('Wrong checkpoint format')
    if legacy:
        engine=r.module(SRC/'ff2_fairness_baselines.py','native_eval_engine')
        if ck['source_hashes']!=engine.hashes(): raise ValueError('Native source changed')
        model=engine.make_model(sources)
    else:
        if ck['source_hashes']!=r.hashes(): raise ValueError('Expanded source changed')
        engine=r.configure(c,fixture=True)
        model=r.zoo.Adapter(c['method'],sources,fixture=True)
    train,val=engine.checked(c,preflight)
    plan=json.loads((HERE/'protocol.json').read_text());test=plan['splits']['test']
    if r.common.sha(test['path'])!=test['sha256']: raise ValueError('Fixed TEST manifest changed')
    rows=r.common.read_manifest(test['path'])
    if len(rows)!=test['rows']: raise ValueError('TEST row count changed')
    for field in ('image','identity_ids','source_ids'):
        fit={v for row in train+val for v in row[field].split(';') if v}
        evaluation={v for row in rows for v in row[field].split(';') if v}
        if fit&evaluation: raise ValueError('Undeclared fit:TEST overlap: '+field)
    fit_hash={row['image_sha256'] for row in train+val}
    for row in rows:
        row['image_sha256']=r.common.sha(row['image'])
        if row['image_sha256'] in fit_hash: raise ValueError('Fit:TEST image byte overlap')
    model.load_state_dict(ck['model'],strict=True);model=model.to(device).eval();del ck
    out=r.common.fresh_dir(out)
    z=engine.predict(model,rows,c,device);threshold=verified['calibration']['threshold_logit']
    scores=dict(rows=rows,logits=list(z),threshold=threshold,min_support=c['min_support'],split='test',
                manifest_sha256=test['sha256'],checkpoint_sha256=before,
                calibration_source='unchanged selected FF2 validation only')
    r.common.write_json(out/'scores_PRIVATE.json',scores)
    receipt=dict(status='EVALUATED_FROZEN',method=c['method'],checkpoint_sha256=before,manifest_sha256=test['sha256'],
                 rows=len(rows),split='test',threshold=threshold,test_threshold_search=False,source_hashes=engine.hashes())
    if before!=r.common.sha(checkpoint):raise ValueError('Checkpoint changed during evaluation')
    r.common.write_json(out/'RUN_VERIFIED.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('run','preflight','sources','out'):p.add_argument('--'+arg,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    evaluate(a.run,a.preflight,a.sources,a.out,torch.device(a.device))
