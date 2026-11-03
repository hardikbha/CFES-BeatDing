"""Exact three-key change, initial-state and optimizer-group verification."""
import importlib.util,json,hashlib
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('plain',HERE.parents[2]/'CES_FL/src/ces_fl_head_ablation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/lh_low_lr3_s17_20261103T140700Z')
c=m.load_config(HERE/'config.json');old=m.load_config(HERE.parent/'plain_head_s17_20261103T101418Z/L_H_plain_hierarchical.json')
diff={k:{'old':old[k],'new':c[k]} for k in c if c[k]!=old[k]}
assert diff==dict(head_lr={'old':.001,'new':.0001},encoder_lr={'old':.00001,'new':.000001},epochs={'old':10,'new':3})
torch.set_num_threads(2);a=m.load_initial(c);b=m.load_initial(old)
assert a.counts()=={'total':20809001,'trainable':6790433}
assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
assert all(not mod.training for mod in a.encoder.modules())
opt=m.build_optimizer(a,c)
assert [g['lr'] for g in opt.param_groups]==[.0001,.000001]
def digest(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().numpy().tobytes())
 return h.hexdigest()
x=dict(status='PASS',changes=diff,source_sha256=m.sha(m.__file__),config_sha256=m.sha(HERE/'config.json'),config_digest=m.digest(c),initial_state_identical_to_original_L_H=True,
 linear_head_state_hash=digest(a.classifier.state_dict()),encoder_state_hash=digest(a.encoder.state_dict()),optimizer_lrs=[g['lr'] for g in opt.param_groups],parameters=a.parameter_report(),encoder_eval=True,test_read=False)
assert x['linear_head_state_hash']=='9600162d7f69931f93d5e9e44ec778cd4e361c9ad0af38b66708508b50d04907'
assert x['encoder_state_hash']=='28390525d92313ec786b694ce21eef17db9c5157e1b89c37a0ac8071a475dbcb'
m.write_json(ROOT/'REQUEST_CHECK.json',x);print(json.dumps(x,indent=2))
