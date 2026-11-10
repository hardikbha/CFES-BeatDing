#!/usr/bin/env python3
"""Use ONLY the synthetic smoke fixture; verify actual masked full-L_H training."""
import argparse,importlib.util,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--smoke-root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
source=Path(__file__).resolve().parents[3]/'src/ces_fl_lh_channel_selection.py'
s=importlib.util.spec_from_file_location('masked_full_test',source);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.set_num_threads(2);root=m.fresh_dir(a.out);base=Path(a.smoke_root)
c=json.loads((base/'H/config.json').read_text())
assert c['synthetic_fixture'] and c['source_sha256']==m.sha(source)
masks,_,_=m.select_masks(np.arange(2048),np.arange(2048),np.arange(2048))
maskfile=root/'MASKS_SYNTHETIC.json'
m.write_json(maskfile,dict(format='TRAIN_CHANNEL_MASK_v1',source_sha256=m.sha(source),context_digest=m.digest(m.context_config(c)),synthetic_fixture=True,validation_scores_used=False,removed=masks))
c['feature_selection'].update(mode='informed',mask_file=str(maskfile.resolve()),mask_sha256=m.sha(maskfile))
m.check_config(c,fixture=True)
before=m.load_initial(c).state_dict()
result=m.train(c,base/'preflight_H',root/'masked_H','D',torch.device('cpu'),fixture=True)
ck=torch.load(root/'masked_H/best.pt',weights_only=True)
removed=masks['informed'];assert int((ck['model']['feature_mask']==0).sum())==41
assert torch.equal(ck['model']['feature_mask'],before['feature_mask'])
assert torch.equal(ck['model']['encoder.conv1.weight'],before['encoder.conv1.weight'])
assert not torch.equal(ck['model']['encoder.conv4.pointwise.weight'],before['encoder.conv4.pointwise.weight'])
assert not torch.equal(ck['model']['classifier.weight'],before['classifier.weight'])
for k,v in before.items():
    if 'running_' in k or 'num_batches_tracked' in k:assert torch.equal(v,ck['model'][k]),k
history=json.loads((root/'masked_H/history.json').read_text())
for r in history:
    assert r['originals_once_each'] and r['original_real_pairs']==4
    for key in ('error_fairness','original_sensitivity','aux_sensitivity','response_alignment','original_control_stability','aux_control_stability'):
        assert key in r['losses_step_mean'] and np.isfinite(r['losses_step_mean'][key])
assert result['status']=='COMPLETED_VERIFIED'
m.write_json(root/'MASKED_LIFECYCLE_PASS.json',dict(status='PASS',source_sha256=m.sha(source),device='cpu',epochs=2,arm='D',removed_channels=41,full_LH_losses_retained=True,early_layers_and_BN_unchanged=True,exit_and_head_updated=True,mask_unchanged=True,native_selected_checkpoint_replay=True,synthetic_only=True,real_TEST_access=False))
print((root/'MASKED_LIFECYCLE_PASS.json').read_text())
