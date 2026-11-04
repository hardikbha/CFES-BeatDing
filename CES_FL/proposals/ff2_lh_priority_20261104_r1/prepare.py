#!/usr/bin/env python3
"""Audited encoder-only transfer from completed FF2 native BCE50; no training."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=load(ROOT/'CES_FL/src/ces_fl_head_ablation.py','ff2_lh_model')

def encoder_state(state):
    model=m.Detector(head_type='linear',policy='exit')
    expected=model.encoder.state_dict()
    if set(state)-set(expected)!={'last_linear.weight','last_linear.bias'}:
        raise ValueError('Unexpected native architecture/head keys')
    encoder={k:state[k] for k in expected}
    model.encoder.load_state_dict(encoder,strict=True)
    return {'encoder.'+k:v.detach().cpu().clone() for k,v in encoder.items()}

def prepare(run,out):
    request=json.loads((HERE/'request.json').read_text())
    if m.sha(m.__file__)!=request['source_sha256']:raise ValueError('L_H scientific source changed')
    run=Path(run);checkpoint=run/'best.pt'
    if not (run/'DONE').is_file():raise ValueError('Native pretraining is incomplete')
    sha=m.sha(checkpoint)
    if sha!=request['native_pretraining_checkpoint_sha256']:raise ValueError('Not the pinned native FF2 BCE50 checkpoint')
    verified=json.loads((run/'RUN_VERIFIED.json').read_text())
    if verified['status']!='COMPLETED_VERIFIED' or verified['epochs']!=50 or verified['method']!='bce' or verified.get('fixture') or verified['checkpoint_sha256']!=sha:
        raise ValueError('Native completion/replay evidence mismatch')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True);c=ck['config']
    if ck['format']!='FF2_COMMON_RECIPE_FAIRNESS_v1' or ck.get('fixture') or c['method']!='bce' or c['epochs']!=50 or ck['selected_epoch']!=40:
        raise ValueError('Wrong pretraining format/selection')
    if c!=json.loads((run/'config.json').read_text()):raise ValueError('Native config changed')
    for split in ('train','val'):
        expected=request['splits'][split]
        if c[split]!=expected['path'] or c[split+'_sha256']!=expected['sha256'] or m.sha(c[split])!=expected['sha256']:
            raise ValueError('FF2 pretraining membership mismatch')
    state=encoder_state(ck['model']);out=m.fresh_dir(out)
    export=out/'encoder_transfer.pt'
    provenance={'source_checkpoint_sha256':sha,'source_run':str(run.resolve()),'native_config':c,
                'completed_epochs':50,'selected_epoch':40,'only_encoder_transferred':True,
                'classifier_not_transferred':True,'native_verification_sha256':m.sha(run/'RUN_VERIFIED.json')}
    # Existing loader compatibility format; explicit transfer provenance prevents
    # representing this as an original CES-FL pretraining run.
    torch.save({'format':'CESFL_CF5050_v1','stage':'pretrain','synthetic_fixture':False,
                'config':{'backbone':'xception','train':c['train'],'val':c['val'],
                          'input_hashes':{k:c[k+'_sha256'] for k in ('train','val')}},
                'model':state,'transfer_provenance':provenance},export)
    bound={}
    for arm in ('L_H','L_A'):
        cfg=json.loads((HERE/(arm+'_template.json')).read_text())
        cfg.update(init_checkpoint=str(export.resolve()),init_sha256=m.sha(export))
        m.check_config(cfg)
        restored=m.load_initial(cfg)
        for key,value in state.items():
            torch.testing.assert_close(restored.encoder.state_dict()[key[len('encoder.'):]],value,rtol=0,atol=0)
        m.write_json(out/(arm+'.json'),cfg);bound[arm]=m.digest(cfg)
    m.write_json(out/'TRANSFER_VERIFIED.json',{'status':'PASS','source_checkpoint_sha256':sha,
        'export_sha256':m.sha(export),'config_digests':bound,'encoder_tensor_equality':True,
        'native_pretraining_recipe_differs_from_original_FFPP_P50':True,'new_training':False,'test_read':False})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-run',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();prepare(a.native_run,a.out)
