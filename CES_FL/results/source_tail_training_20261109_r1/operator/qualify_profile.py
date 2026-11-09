"""Allocated-device synthetic checks and fixed-recipe TRAIN timing; no TEST/CAL."""
import argparse,importlib.util,json,time,tempfile
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root',required=True);a=p.parse_args()
src=Path(a.source);out=Path(a.root);prop=src/'CES_FL/proposals/source_tail_training_20261109_r1'
spec=importlib.util.spec_from_file_location('st',prop/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
device=torch.device('cuda');torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m.core.seed_all(17)
assert torch.cuda.is_available() and torch.cuda.device_count()==1
binding={'device_count':1,'device_name':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'visible':__import__('os').environ['CUDA_VISIBLE_DEVICES']}
(out/'CUDA_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
# Supplied synthetic cache factory, with the same pinned train API on CUDA.
sp=importlib.util.spec_from_file_location('checks',prop/'check_pipeline.py');checks=importlib.util.module_from_spec(sp);sp.loader.exec_module(checks)
with tempfile.TemporaryDirectory(dir=out,prefix='cuda_fixture_') as td:
    td=Path(td);checks.cache(td/'cache');fixture=m.config();fixture.update(batch_size=8,min_support=2,min_auc=0.,max_fnr=1.,auxiliary_per_source=4)
    for arm in ('control','treatment'):
        r=m.train(fixture,td/'cache',td/arm,arm,device,fixture=True)
        assert r['status']=='COMPLETED_VERIFIED' and r['epochs']==3
    model=m.core.Detector(head_type='linear',policy='frozen').to(device).eval();x=torch.randn(2,3,224,224,device=device)
    with torch.inference_mode():
        f=model.encoder.features(x).mean((2,3));torch.testing.assert_close(model.classifier(f).squeeze(1),model(x)['logit'],rtol=0,atol=0)
    assert not any(q.requires_grad for q in model.encoder.parameters())
    del model,x
torch.cuda.empty_cache()
(out/'CUDA_CHECKS.json').write_text(json.dumps({'status':'PASS','fixture_only':True,'source_hashes':m.hashes(),
    'checks':['both pinned three-epoch synthetic arms on CUDA','finite head gradients and selected-score replay','full224 encoder-feature/native-forward identity','frozen encoder']},indent=2)+'\n')
# No benchmark metric is inferred from this timing profile. Exact TRAIN images,
# teacher, preprocessing, SHA checks, batch and worker count match prepare().
c=m.config();teacher=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_lh_ps_20261106_r1/L_A')
model,ck=m.core.load_trained(teacher/'best.pt',torch.device('cpu'));model.requires_grad_(False);model.eval()
assert m.core.sha(teacher/'best.pt')==c['teacher_checkpoint_sha256']
rows=m.core.read_manifest(c['train']['path']);inventory={r['image']:r['image_sha256'] for r in ck['fit_membership']}
ix=np.random.default_rng(17).choice(len(rows),1024,replace=False);sample=[dict(rows[int(i)],image_sha256=inventory[rows[int(i)]['image']]) for i in ix]
model=model.to(device).eval();loader=m.DataLoader(m.Images(sample),batch_size=c['batch_size'],num_workers=c['workers'])
durations=[];start=time.perf_counter();previous=start;done=0
with torch.inference_mode():
    for x,indices in loader:
        x=x.to(device);f=model.encoder.features(x).mean((2,3));z=model.classifier(f).squeeze(1)
        assert torch.isfinite(z).all();torch.cuda.synchronize();t=time.perf_counter();durations.append(t-previous);previous=t;done+=len(indices)
elapsed=time.perf_counter()-start
# Micro-step timing includes actual2048-wide head and both objective paths,
# disposable only; production still begins at the unchanged teacher head.
head=torch.nn.Linear(2048,1).to(device);head.load_state_dict(model.classifier.state_dict());opt=torch.optim.AdamW(head.parameters(),lr=c['head_lr'],weight_decay=c['weight_decay'])
ref=m.objective.fit_train_reference(torch.linspace(-5,5,100,device=device));lossfn=m.objective.SourceTailObjective(ref).to(device)
batch=torch.randn(32,2048,device=device);labels=torch.tensor([0,1]*16,device=device);teacher_logits=head(batch).squeeze(1).detach();aux=torch.randn(32,2048,device=device);sources=torch.tensor([0]*16+[1]*16,device=device)
torch.cuda.synchronize();st=time.perf_counter()
for _ in range(100):
    loss=lossfn(head(batch).squeeze(1),labels,teacher_logits,head(aux).squeeze(1),sources)['total'];opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),c['clip_norm']);opt.step()
torch.cuda.synchronize();step=(time.perf_counter()-st)/100
rate=done/elapsed;images=c['train']['rows']+3*c['val']['rows'] # shared cache and two native exports
estimated=images/rate + 2*c['epochs']*int(np.ceil(c['train']['rows']/32))*step
estimated=1.4*estimated+180 # I/O/hash/head-validation/bookend margin, fixed before outcome
result={'status':'PASS' if estimated/3600<1.7 else 'BUDGET_PENDING','profile_train_rows':done,'batches':len(durations),
    'batch_size':c['batch_size'],'workers':c['workers'],'wall_seconds_including_io':elapsed,'images_per_second':rate,
    'batch_seconds':durations,'estimated_two_arm_head_step_seconds':step,'projected_remaining_total_gpu_hours_with_margin':estimated/3600,
    'estimated_float32_feature_bytes':(c['train']['rows']+c['val']['rows'])*2048*4,
    'class_coverage':{str(k):sum(r['label']==k for r in sample) for k in (0,1)},
    'fake_family_coverage':{family:sum(r['family']==family and r['label']==1 for r in sample) for family in sorted({r['family'] for r in sample if r['label']==1})},
    'source_hashes':m.hashes(),'no_teacher_or_encoder_updates':True,'selected_from_TRAIN_seed':17,'test_cal_read':False}
(out/'REAL_TRAIN_PROFILE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='batch_seconds'}),flush=True)
if result['status']!='PASS':raise SystemExit(3)
