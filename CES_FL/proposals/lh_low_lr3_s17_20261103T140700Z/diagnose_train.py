"""Read only old L_H checkpoints; fixed TRAIN images and epoch0 procedural views."""
import argparse,importlib.util,json,math
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[2]
spec=importlib.util.spec_from_file_location('plain_model',SOURCE/'CES_FL/src/ces_fl_head_ablation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OLD=Path('/home/sachin.chaudhary/ces_fl_runs/plain_head_s17_20261103T101418Z/L_H')

def stats(values):
 a=np.asarray(values,dtype=float)
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),p95=float(np.quantile(a,.95)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
 out=m.fresh_dir(a.out);torch.set_num_threads(2);device=torch.device(a.device)
 c=m.load_config(HERE/'config.json');old=json.loads((OLD/'config.json').read_text())
 assert {k for k in c if c[k]!=old[k]}=={'head_lr','encoder_lr','epochs'}
 rows=m.read_manifest(c['train']);active,_=m.group_inventory(rows,old);chosen=set()
 for ax,groups in active.items():
  for g,cell in groups.items():chosen.add(min(cell[0],key=lambda i:m.seed_for(c['seed'],ax,g,rows[i]['image'])))
 for label in (0,1):
  pool=sorted([i for i,r in enumerate(rows) if r['label']==label],key=lambda i:m.seed_for(17,'fixed-train-drift',rows[i]['image']))
  for i in pool:
   if sum(rows[j]['label']==label for j in chosen)>=32:break
   chosen.add(i)
 indices=sorted(chosen);assert len(indices)==64 and sum(rows[i]['label']==0 for i in indices)==32
 # Views are generated once with the unchanged ORIGINALS(epoch=0) path.
 dataset=m.Originals(rows,old,epoch=0,pairs=True);samples=[dataset[i] for i in indices]
 values={};hashes={};epochs={};buffers_ok={}
 for name in ('best','last'):
  path=OLD/(name+'.pt');hashes[name]=m.sha(path);ck=torch.load(path,map_location='cpu',weights_only=True)
  assert ck['source_sha256']==m.sha(m.__file__) and ck['config']==old and ck['arm']=='D' and not ck['synthetic_fixture']
  epochs[name]=ck['epoch'];model=m.Detector(old['hidden'],old['gate_hidden'],old['encoder_policy'],old['head_type'])
  model.load_state_dict(ck['model'],strict=True);model.to(device).eval()
  before={k:v.clone() for k,v in model.named_buffers()};pred={k:[] for k in ('x','s','t')}
  with torch.inference_mode():
   for start in range(0,len(samples),16):
    for key in pred:pred[key].extend(model(torch.stack([s[key] for s in samples[start:start+16]]).to(device))['logit'].float().cpu().tolist())
  assert all(torch.isfinite(torch.tensor(v)).all() for v in pred.values())
  buffers_ok[name]=all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in before.items());assert buffers_ok[name]
  assert m.sha(path)==hashes[name];values[name]=pred;del model,ck
 perrow=[]
 for j,i in enumerate(indices):
  row=rows[i];v=dict(image=row['image'],image_sha256=m.sha(row['image']),label=row['label'],**{ax:row[ax] for ax in m.AXES})
  for name in values:
   z,zs,zt=(values[name][key][j] for key in ('x','s','t'))
   v[name]=dict(original_logit=z,sham_logit=zs,treated_logit=zt,raw_sham_abs=abs(zs-z),transformed_sham_abs=abs(math.asinh(zs)-math.asinh(z)),asinh_derivative_at_original=1/math.sqrt(1+z*z))
  v['last_minus_best_original_logit']=v['last']['original_logit']-v['best']['original_logit'];perrow.append(v)
 aggregate={}
 for name in values:
  aggregate[name]=dict(epoch=epochs[name],checkpoint_sha256=hashes[name],by_label={})
  for label in (0,1):
   rr=[v for v in perrow if v['label']==label];aggregate[name]['by_label'][str(label)]={key:stats([r[name][key] for r in rr]) for key in ('original_logit','asinh_derivative_at_original')}
  real=[v for v in perrow if v['label']==0]
  aggregate[name]['real_sham_consistency']={k:stats([r[name][k] for r in real]) for k in ('raw_sham_abs','transformed_sham_abs')}
  aggregate[name]['real_sham_consistency']['ratio_sum_transformed_to_raw']=sum(r[name]['transformed_sham_abs'] for r in real)/max(1e-15,sum(r[name]['raw_sham_abs'] for r in real))
 groups={}
 for ax in c['axes']:
  groups[ax]={g:{'n':sum(v[ax]==g for v in perrow),'last_minus_best_original_logit':stats([v['last_minus_best_original_logit'] for v in perrow if v[ax]==g])} for g in sorted({v[ax] for v in perrow})}
 m.write_json(out/'fixed_train_scores_PRIVATE.json',perrow)
 result=dict(status='PASS',device=a.device,source_sha256=m.sha(m.__file__),config_digest=m.digest(c),train_sha256=c['train_sha256'],sample_size=64,real_samples=32,fake_samples=32,
  sample_selection='Same deterministic stratified review real representatives plus seed17 hash-ranked TRAIN rows to32/class; fixed epoch0 views shared by best/last',
  checkpoints=aggregate,by_axis=groups,checkpoint_BN_buffers_unchanged=buffers_ok,no_gradient_updates=True,val_or_test_read=False,
  interpretation='asinh derivative1/sqrt(1+z^2) shrinks absolute sham differences at large logit magnitude. This audit measures drift; it does not establish its causal origin or justify changing the loss.')
 m.write_json(out/'TRAIN_DRIFT.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
