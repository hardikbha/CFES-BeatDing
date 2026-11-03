"""Fresh configuration checks and exact-byte procedural-review reuse, no new approvals."""
import argparse,copy,hashlib,importlib.util,json
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent; SOURCE=HERE.parents[2]
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/lh_sham_noalign_s17_20261103T165600Z')
CONTROL=Path('/home/sachin.chaudhary/ces_fl_runs/lh_low_lr3_s17_20261103T140700Z')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('candidate',SOURCE/'CES_FL/src/ces_fl_head_ablation.py')
old=load('control','/home/sachin.chaudhary/ces_fl_sources/cesfl-lh-low-lr3-execution-20261103/CES_FL/src/ces_fl_head_ablation.py')
guard=load('review_guard',HERE.parent/'plain_head_s17_20261103T101418Z/reuse_review.py')
def read(p):return json.loads(Path(p).read_text())
def verify(arm):
 c=m.load_config(HERE/(arm+'.json')); previous=old.load_config(CONTROL/'training/config.json')
 expected=dict(previous,source_sha256=m.sha(m.__file__))
 if arm=='SHAM_FIX':expected['sham_transform']='asinh_of_difference'
 else:expected['lambda_alignment']=0.
 assert c==expected
 torch.set_num_threads(2);a=m.load_initial(c);b=old.load_initial(previous)
 assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
 def digest(state):
  h=hashlib.sha256()
  for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().numpy().tobytes())
  return h.hexdigest()
 assert digest(a.classifier.state_dict())=='9600162d7f69931f93d5e9e44ec778cd4e361c9ad0af38b66708508b50d04907'
 assert digest(a.encoder.state_dict())=='28390525d92313ec786b694ce21eef17db9c5157e1b89c37a0ac8071a475dbcb'
 assert [g['lr'] for g in m.build_optimizer(a,c).param_groups]==[1e-4,1e-6]
 a.train();assert all(not mod.training for mod in a.encoder.modules())
 assert (CONTROL/'training/DONE').is_file()
 r=read(CONTROL/'training/RUN_VERIFIED.json')
 assert r['selected_checkpoint_sha256']==m.sha(CONTROL/'training/best.pt')=='59143159eea223d97bc436b510f13ca90b750f00851071a860f91ec9231d4a06'
 assert r['epochs_completed']==3 and r['source_sha256']==old.sha(old.__file__)
 m.write_json(ROOT/arm/'REQUEST_CHECK.json',dict(status='PASS',arm=arm,source_sha256=m.sha(m.__file__),config_sha256=m.sha(HERE/(arm+'.json')),config_digest=m.digest(c),identical_fresh_initialization=True,head_hash=digest(a.classifier.state_dict()),encoder_hash=digest(a.encoder.state_dict()),parameters=a.parameter_report(),no_test_access=True))
 print('REQUEST_CHECK_PASS',arm,flush=True)
def review(arm):
 c=m.load_config(HERE/(arm+'.json'));previous=old.load_config(CONTROL/'training/config.json')
 oq=CONTROL/'review/qualification.json';old.check_qualification(previous,oq)
 q=read(oq);prior=read(q['preview_file']);pv=ROOT/arm/'preview/preview_PRIVATE.json';new=read(pv)
 assert new['source_sha256']==m.sha(m.__file__) and new['config_digest']==m.digest(c)
 guard.equivalent_cases(prior,new)
 copied=copy.deepcopy(read(q['review_file']));assert copied['preview_sha256']==q['preview_sha256']
 copied['preview_sha256']=m.sha(pv)
 copied['reuse_provenance']=dict(qualification_path=str(oq),qualification_sha256=m.sha(oq),basis='Exact case IDs/settings/image and sheet bytes unchanged; all named AI decisions preserved verbatim',new_visual_review_performed=False)
 out=m.fresh_dir(ROOT/arm/'review');m.write_json(out/'review_PRIVATE.json',copied)
 m.qualify(pv.parent,out/'review_PRIVATE.json',out/'qualification.json')
 print('REVIEW_IDENTICAL_BYTES_PASS',arm,len(copied['cases']),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['verify','review']);p.add_argument('arm',choices=['SHAM_FIX','NO_ALIGN']);a=p.parse_args();globals()[a.stage](a.arm)
