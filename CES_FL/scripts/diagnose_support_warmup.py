"""CES-FL TRAIN gradient diagnostics and frozen VAL replay only; no Ding/TEST."""
import argparse,importlib.util,json,math
from pathlib import Path
import numpy as np
import torch

def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 source=Path(a.source);root=Path(a.run);out=Path(a.out);out.mkdir(exist_ok=False)
 m=module(source/'CES_FL/src/ces_fl_forensic_response.py','diagnostic_cesfl');torch.set_num_threads(2);device=torch.device('cuda');m.seed_all(17)
 c=m.load_config(root/'E_support/config.json');rows,val,_=m.checked_inputs(c,root/'E_support/preflight');active,support=m.group_inventory(rows,c)
 weights=m.original_weights(rows,c['balance_axis']);b=next(iter(m.make_loader(rows,c,epoch=0,pairs=True,shuffle=True)))
 summaries=[]
 for initialization in ['fresh_p50_heads','historical_selected_CESFL_D']:
  model=m.load_initial(c).to(device)
  old_path=Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_overlapfix_20261103_r1_ffpp_s17/D/best.pt');before=m.sha(old_path)
  if initialization.startswith('historical'):
   ck=torch.load(old_path,map_location='cpu',weights_only=True)
   assert ck['source_sha256']=='89530025a00b9b0405fba7026d3eed5f2749d86620174ad3ffde93a44f8b770b';model.load_state_dict(ck['model'],strict=True);del ck
  real=[r for r in rows if r['label']==0];real=sorted(real,key=lambda r:m.seed_for(17,'train-calibration',r['image']))[:c['train_calibration_size']]
  threshold=m.calibrate(m.predict(model,real,c,device),c['target_fpr'])['threshold_logit'];model.train()
  params=[p for p in model.parameters() if p.requires_grad]
  for event,axis in enumerate(c['axes']):
   aux=m.auxiliary(rows,active,c,event,0);assert aux['axis']==axis
   loss,_=m.train_step(model,b,aux,weights,c,'D',threshold,device,0)
   terms={'classification':loss['classification'],'error_control':c['aux_every']*c['lambda_fair']*loss['error_fairness'],
    'sensitivity':c['lambda_sensitivity']*(loss['original_sensitivity']+c['aux_every']*loss['aux_sensitivity']),
    'processing_control':c['lambda_sensitivity']*c['sham_weight']*(loss['original_control_stability']+c['aux_every']*loss['aux_control_stability']),
    'alignment':c['aux_every']*c['lambda_alignment']*loss['response_alignment']}
   gradients={}
   for key,value in terms.items():
    pieces=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
    gradients[key]=torch.cat([(torch.zeros_like(p) if g is None else g).float().flatten() for p,g in zip(params,pieces)])
    assert torch.isfinite(gradients[key]).all()
   norms={k:float(torch.linalg.vector_norm(v)) for k,v in gradients.items()};cos={}
   for key,g in gradients.items():
    denom=norms['classification']*norms[key];cos[key]=float(torch.dot(gradients['classification'],g)/denom) if denom>0 else None
   summaries.append(dict(initialization=initialization,axis=axis,weighted_loss={k:float(v.detach()) for k,v in terms.items()},weighted_gradient_norm=norms,cosine_with_classification=cos,train_support_fraction=aux['support_fraction'].tolist(),sampled_rows=len(aux['indices']),optimizer_steps=0))
   del gradients,loss,terms
  assert m.sha(old_path)==before
  del model
  torch.cuda.empty_cache()
 m.write_json(out/'TRAIN_GRADIENTS.json',dict(source_sha256=m.sha(m.__file__),status='PASS',rows_from='TRAIN only',validation_gradients=False,test_read=False,optimizer_steps=0,diagnostics=summaries,warning='Gradient cosines are snapshots, not causal evidence or permission to tune hyperparameters'))
 # Existing selected CES-FL controls, not Ding, replay only on their frozen VAL.
 old=module(Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-forensic-response-c332fd34/CES_FL/src/ces_fl_forensic_response.py'),'historical_cesfl')
 replay=[]
 for arm in 'ACD':
  path=Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_overlapfix_20261103_r1_ffpp_s17')/arm/'best.pt';before=m.sha(path)
  model,ck=old.load_trained(path,device);assert ck['config']['val']==c['val'] and ck['config']['val_sha256']==c['val_sha256']
  z=old.predict(model,val,ck['config'],device);threshold=ck['calibration']['threshold_logit'];report=old.metric_report(val,z,threshold,ck['config']['min_eval_support']);assert report==ck['validation']
  m.write_json(out/f'{arm}_validation_scores_PRIVATE.json',dict(rows=val,logits=z.tolist(),threshold=threshold,min_support=c['min_eval_support'],manifest_sha256=c['val_sha256'],checkpoint_sha256=before,split='validation',test_read=False,calibration_source='unchanged selected checkpoint validation only'))
  replay.append(dict(arm=arm,checkpoint_sha256=before,selected_epoch=ck['epoch'],report=report,threshold=threshold));assert m.sha(path)==before
  del model,ck;torch.cuda.empty_cache()
 m.write_json(out/'CONTROL_VALIDATION_REPLAY.json',dict(status='PASS',controls='existing CES-FL only; no Ding model computation',test_read=False,source_sha256=m.sha(m.__file__),replayed=replay))
 print(json.dumps(dict(status='PASS',gradient_snapshots=len(summaries),controls_replayed=3,test_read=False,optimizer_steps=0)),flush=True)
if __name__=='__main__':main()
