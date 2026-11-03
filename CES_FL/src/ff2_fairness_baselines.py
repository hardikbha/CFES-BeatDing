#!/usr/bin/env python3
"""FF2 common-recipe Xception BCE/DAG-FDD/DAW-FDD, seed17.

Official pinned Xception classes and official-equivalent CVaR losses; new
auditable training adapter. This is a disclosed common-recipe adaptation,
not a claim to reproduce every original paper training detail. Full encoder
and BN train. No CLIP, CES gates, generated images, GDA or TEST command.
One GPU per independent method; never shard CVaR across independent ranks.
"""
import argparse
import ast
from collections import Counter
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parents[2]
PROPOSAL=ROOT/'CES_FL/proposals/ff2_baseline_suite_s17_20261103T110324Z'
VERSION='FF2_COMMON_RECIPE_FAIRNESS_v1'
METHODS=('bce','dag','daw')


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


common=module(Path(__file__).with_name('ces_fl_head_ablation.py'),'baseline_common')
losses=module(ROOT/'CLIPFAIR/cfes/clipfair_losses.py','official_equivalent_losses')


def hashes():
    paths=[Path(__file__),Path(common.__file__),Path(losses.__file__),PROPOSAL/'upstreams.lock.json']
    return {str(p.relative_to(ROOT)):common.sha(p) for p in paths}


def load_xception(upstream):
    """Execute only the three pinned architecture classes, without registry imports."""
    lock=json.loads((PROPOSAL/'upstreams.lock.json').read_text())['repositories'][0]
    relative='training/networks/xception.py';p=Path(upstream)/relative;data=p.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if blob!=lock['files'][relative]['git_blob']:raise ValueError('Unpinned upstream Xception')
    nodes=[]
    for node in ast.parse(data).body:
        if isinstance(node,ast.ClassDef) and node.name in ('SeparableConv2d','Block','Xception'):
            node=copy.deepcopy(node);node.decorator_list=[];nodes.append(node)
    if [n.name for n in nodes]!=['SeparableConv2d','Block','Xception']:raise ValueError('Architecture classes changed')
    namespace={'torch':torch,'nn':nn,'F':F,'math':math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),namespace)
    return namespace['Xception']


def make_model(upstream,weights=None):
    model=load_xception(upstream)({'mode':'original','num_classes':1,'inc':3,'dropout':False})
    # Official mode=original never executes this adapter. Remove unused parameters.
    model.adjust_channel=nn.Identity()
    if weights:
        state=torch.load(weights,map_location='cpu',weights_only=True)
        state={k:(v[...,None,None] if 'pointwise' in k and v.ndim==2 else v) for k,v in state.items() if 'fc' not in k}
        r=model.load_state_dict(state,strict=False)
        if r.unexpected_keys or set(r.missing_keys)!={'last_linear.weight','last_linear.bias'}:
            raise ValueError(('Unexpected pretrained initialization',r))
    return model


def objective(z,y,g,method):
    if method=='bce':return F.binary_cross_entropy_with_logits(z,y.float())
    if method=='dag':return losses.dag_fdd_loss(z,y,alpha=.5)[0]
    if method=='daw':return losses.daw_fdd_loss(z,y,g,inner_alpha=.9,outer_alpha=.5)[0]
    raise ValueError(method)


def config(path):
    c=json.loads(Path(path).read_text())
    if c['format']!=VERSION or c['method'] not in METHODS or c['source_hashes']!=hashes():raise ValueError('Config/source mismatch')
    fixed=dict(seed=17,epochs=50,batch_size=128,validation_batch=128,lr=.001,momentum=.9,weight_decay=.005,
               target_fpr=.1,min_support=20,size=224,precision='float32',train_bn=True)
    if any(c.get(k)!=v for k,v in fixed.items()):raise ValueError('Fixed common recipe changed; create new proposal')
    for split in ('train','val','native_train','weights'):
        if common.sha(c[split])!=c[split+'_sha256']:raise ValueError('Input hash changed: '+split)
    return c


def read_inputs(c):
    import csv
    train,val=common.read_manifest(c['train']),common.read_manifest(c['val'])
    if len(train)!=c['train_count'] or len(val)!=c['val_count']:raise ValueError('Membership count changed')
    for field in ('image','identity_ids','source_ids'):
        a={v for r in train for v in r[field].split(';') if v};b={v for r in val for v in r[field].split(';') if v}
        if a&b:raise ValueError('Undeclared TRAIN/VAL '+field+' overlap')
    with Path(c['native_train']).open(newline='') as f:native=list(csv.DictReader(f))
    lookup={r['img_path']:r for r in native}
    if len(lookup)!=len(native) or {(r['image'],r['label']) for r in train}!={(r['img_path'],int(r['label'])) for r in native}:
        raise ValueError('Native/canonical TRAIN mismatch')
    mapping={}
    for r in train:
        r['native_group']=int(lookup[r['image']]['intersec_label'])
        if not 0<=r['native_group']<8:raise ValueError('Invalid native group')
        mapping.setdefault(r['native_group'],set()).add(r['gender_race'])
    if len(mapping)!=8 or any(len(v)!=1 for v in mapping.values()) or len(set(next(iter(v)) for v in mapping.values()))!=8:
        raise ValueError('Demographic mapping changed')
    for r in val:r['native_group']=-1
    if any(not any(r['native_group']==g and r['label']==y for r in train) for g in range(8) for y in (0,1)):
        raise ValueError('Missing TRAIN class/group support')
    return train,val


def preflight(c,out):
    out=common.fresh_dir(out);train,val=read_inputs(c);inventory=[];seen={}
    for split,rows in [('train',train),('val',val)]:
        for r in rows:
            data=Path(r['image']).read_bytes();h=hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as im:im.convert('RGB').load()
            if h in seen and seen[h]!=split:raise ValueError('Cross-split byte overlap')
            seen[h]=split;inventory.append({'split':split,'image':r['image'],'sha256':h})
    common.write_json(out/'inventory_PRIVATE.json',inventory)
    common.write_json(out/'PREFLIGHT.json',dict(status='PASS',source_hashes=hashes(),config_digest=common.digest(c),
        inventory_sha256=common.sha(out/'inventory_PRIVATE.json'),train_count=len(train),val_count=len(val),test_read=False))


def checked(c,path):
    root=Path(path);r=json.loads((root/'PREFLIGHT.json').read_text())
    if r['status']!='PASS' or r['source_hashes']!=hashes() or r['config_digest']!=common.digest(c) or r['inventory_sha256']!=common.sha(root/'inventory_PRIVATE.json'):
        raise ValueError('Stale input receipt')
    train,val=read_inputs(c);lookup={(r['split'],r['image']):r['sha256'] for r in json.loads((root/'inventory_PRIVATE.json').read_text())}
    for split,rows in [('train',train),('val',val)]:
        for row in rows:row['image_sha256']=lookup[split,row['image']]
    return train,val


class Images(Dataset):
    def __init__(self,rows,size=224,evaluation=False):self.rows,self.size,self.evaluation=rows,size,evaluation
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index];data=Path(r['image']).read_bytes()
        if r.get('image_sha256') and hashlib.sha256(data).hexdigest()!=r['image_sha256']:raise ValueError('Image changed')
        with Image.open(io.BytesIO(data)) as im:
            im=im.convert('RGB')
            if self.evaluation:
                w,h=im.size;s=self.size
                shape=(s,int(s*h/w)) if w<=h else (int(s*w/h),s)
                im=im.resize(shape,Image.Resampling.BILINEAR)
                left=int(round((shape[0]-s)/2.));top=int(round((shape[1]-s)/2.))
                im=im.crop((left,top,left+s,top+s))
            else:im=im.resize((self.size,self.size),Image.Resampling.BILINEAR)
            x=torch.from_numpy(np.asarray(im,dtype=np.float32).copy()/255.).permute(2,0,1)
        return (x-.5)/.5,r['label'],r.get('native_group',-1),index


@torch.inference_mode()
def predict(model,rows,c,device):
    model.eval();scores=[]
    for x,_,_,_ in DataLoader(Images(rows,c['size'],True),batch_size=c['validation_batch'],num_workers=c['workers']):
        scores.extend(model(x.to(device))[0].squeeze(1).cpu().tolist())
    if len(scores)!=len(rows) or not np.isfinite(scores).all():raise ValueError('Invalid validation logits')
    return scores


def setup(seed):
    common.seed_all(seed);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def optimizer(model,c):return torch.optim.SGD(model.parameters(),lr=c['lr'],momentum=c['momentum'],weight_decay=c['weight_decay'])


def step(model,b,opt,method,device):
    x,y,g,_=b;opt.zero_grad(set_to_none=True)
    z=model(x.to(device))[0].squeeze(1);loss=objective(z,y.to(device),g.to(device),method)
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward()
    gradients=[p.grad for p in model.parameters()]
    if any(g is None for g in gradients) or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
        raise ValueError('Missing/nonfinite model gradients')
    opt.step();return float(loss.detach())


def train(c,rows,val,upstream,out,device,resume=False,fixture=False,stop_steps=None,budget_hours=None):
    """Deterministic mid-epoch resume, one GPU, no stochastic image augmentation."""
    setup(c['seed']);out=Path(out);start=time.monotonic();base=0.;state=None
    if resume:
        if (out/'DONE').exists():raise ValueError('Already completed')
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if state['config']!=c or state['source_hashes']!=hashes() or state['fixture']!=fixture:raise ValueError('Resume mismatch')
        if state.get('best_sha256') and common.sha(out/'best.pt')!=state['best_sha256']:raise ValueError('Selected checkpoint changed')
        base=state['elapsed_seconds']
    else:common.fresh_dir(out)
    allowance=c['max_gpu_hours'] if budget_hours is None else budget_hours
    if not math.isfinite(allowance) or not 0<allowance<=72 or base>=allowance*3600:
        raise ValueError('Positive remaining cumulative execution allowance required; account it in the shared ledger')
    model=make_model(upstream,None if fixture else c['weights']).to(device);opt=optimizer(model,c)
    history=[];best=-math.inf;epoch=0;cursor=0;loss_sum=0.;steps=0;best_sha=None
    if state:
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
        history,best,epoch,cursor,loss_sum,steps,best_sha=(state[k] for k in ('history','best_auc','epoch','cursor','loss_sum','steps','best_sha256'))
    stopping=[False]
    def mark_stop(*args):stopping[0]=True
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    for s in previous:signal.signal(s,mark_stop)
    def snapshot():
        return dict(format=VERSION,config=c,source_hashes=hashes(),fixture=fixture,model=model.state_dict(),optimizer=opt.state_dict(),
                    history=history,best_auc=best,epoch=epoch,cursor=cursor,loss_sum=loss_sum,steps=steps,
                    best_sha256=best_sha,elapsed_seconds=base+time.monotonic()-start)
    common.write_json(out/'config.json',c)
    try:
        while epoch<c['epochs']:
            order=np.random.default_rng(c['seed']+epoch).permutation(len(rows)).tolist()
            batches=[order[i:i+c['batch_size']] for i in range(0,len(order),c['batch_size'])]
            model.train();loader=DataLoader(Images(rows,c['size']),batch_sampler=batches[cursor:],num_workers=c['workers'])
            for batch in loader:
                if batch[3].tolist()!=batches[cursor]:raise ValueError('Original exposure/order mismatch')
                loss_sum+=step(model,batch,opt,c['method'],device);cursor+=1;steps+=1
                elapsed=base+time.monotonic()-start
                if stopping[0] or elapsed>=allowance*3600 or (stop_steps and steps>=stop_steps):
                    common.atomic_save(out/'last.pt',snapshot())
                    common.write_json(out/'STOPPED.json',dict(status='INCOMPLETE_RESUMABLE',epoch=epoch+1,next_batch=cursor,steps=steps,elapsed_seconds=elapsed))
                    return {'status':'INCOMPLETE_RESUMABLE'}
                if steps%500==0:common.atomic_save(out/'last.pt',snapshot())
            z=predict(model,val,c,device);y=np.array([r['label'] for r in val]);cal=common.calibrate(np.array(z)[y==0],c['target_fpr'])
            report=common.metric_report(val,z,cal['threshold_logit'],c['min_support']);auc=report['overall']['auc']
            record=dict(epoch=epoch+1,steps=steps,originals_once_each=True,original_count=len(rows),mean_loss=loss_sum/len(batches),validation=report,calibration=cal)
            history.append(record)
            if auc>best:
                best=auc;ck=snapshot();ck.update(selected_epoch=epoch+1,calibration=cal,validation=report)
                common.atomic_save(out/'best.pt',ck);best_sha=common.sha(out/'best.pt')
            epoch+=1;cursor=0;loss_sum=0.
            common.atomic_save(out/'last.pt',snapshot());common.write_json(out/'history.json',history)
            print(json.dumps({'method':c['method'],'epoch':epoch,'val_auc':auc,'elapsed_seconds':base+time.monotonic()-start}),flush=True)
        ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        del model,opt
        model=make_model(upstream).to(device);model.load_state_dict(ck['model'],strict=True)
        z=predict(model,val,c,device);report=common.metric_report(val,z,ck['calibration']['threshold_logit'],c['min_support'])
        if report!=ck['validation']:raise ValueError('Selected validation replay mismatch')
        result=dict(status='COMPLETED_VERIFIED',method=c['method'],epochs=epoch,selected_epoch=ck['selected_epoch'],
                    checkpoint_sha256=common.sha(out/'best.pt'),source_hashes=hashes(),validation=report,
                    calibration=ck['calibration'],test_read=False,fixture=fixture,elapsed_seconds=base+time.monotonic()-start,
                    fixed_probability_0_5=common.metric_report(val,z,0.,c['min_support']))
        common.write_json(out/'RUN_VERIFIED.json',result)
        common.write_json(out/'validation_scores_PRIVATE.json',dict(rows=val,logits=z,threshold=ck['calibration']['threshold_logit'],
            min_support=c['min_support'],manifest_sha256=c['val_sha256'],checkpoint_sha256=result['checkpoint_sha256'],split='validation'))
        (out/'DONE').write_text('Selected checkpoint independently replayed on VAL; TEST not read.\n')
        if (out/'STOPPED.json').exists():(out/'STOPPED.json').rename(out/'PRIOR_STOP_RESUMED.json')
        return result
    finally:
        for s,handler in previous.items():signal.signal(s,handler)


def dry_run(c,rows,upstream,out,device,steps):
    out=common.fresh_dir(out);setup(c['seed']);model=make_model(upstream,c['weights']).to(device).train();opt=optimizer(model,c)
    loader=DataLoader(Images(rows,c['size']),batch_size=c['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(c['seed']),num_workers=c['workers'])
    times=[];cycles=[];last=time.monotonic()
    for i,b in enumerate(loader):
        if i==steps:break
        if device.type=='cuda':torch.cuda.synchronize()
        t=time.monotonic();value=step(model,b,opt,c['method'],device)
        if device.type=='cuda':torch.cuda.synchronize()
        now=time.monotonic();times.append(now-t);cycles.append(now-last);last=now
    record=dict(status='PASS',source_hashes=hashes(),config_digest=common.digest(c),device=str(device),steps=len(times),
        method=c['method'],last_loss=value,median_step_seconds=float(np.median(times[2:] or times)),
        median_cycle_including_input_seconds=float(np.median(cycles[2:] or cycles)),
        estimated_training_only_gpu_hours=float(np.median(cycles[2:] or cycles))*math.ceil(len(rows)/c['batch_size'])*c['epochs']/3600,
        estimate_excludes_validation_checkpoints=True,parameters=sum(p.numel() for p in model.parameters()),test_read=False)
    common.write_json(out/'DRY_RUN.json',record);print(json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','dry-run','train'])
    p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--upstream')
    p.add_argument('--preflight');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--resume',action='store_true')
    p.add_argument('--receipt');p.add_argument('--software-receipt');p.add_argument('--steps',type=int,default=20)
    p.add_argument('--budget-hours',type=float,help='Cumulative one-GPU execution allowance; scheduling only, never changes scientific config')
    a=p.parse_args();c=config(a.config)
    if a.command=='preflight':preflight(c,a.out);return
    if not a.upstream or not a.preflight:p.error('--upstream and --preflight required')
    device=torch.device(a.device)
    if device.type=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):raise ValueError('Bind exactly one allocated GPU per method')
    rows,val=checked(c,a.preflight)
    if a.command=='dry-run':
        if a.steps<3:p.error('At least three backward steps required')
        dry_run(c,rows,a.upstream,a.out,device,a.steps);return
    if not a.receipt:p.error('Training requires source/config/device-bound DRY_RUN receipt')
    r=json.loads(Path(a.receipt).read_text())
    if r['status']!='PASS' or r['source_hashes']!=hashes() or r['config_digest']!=common.digest(c) or r['device']!=str(device):raise ValueError('Stale backward receipt')
    if not a.software_receipt:p.error('Training requires a fresh synthetic integration receipt')
    s=json.loads(Path(a.software_receipt).read_text())
    if s.get('status')!='PASS' or s.get('source_hashes')!=hashes() or s.get('device')!=str(device) or s.get('fixture_only') is not True:
        raise ValueError('Stale software receipt')
    result=train(c,rows,val,a.upstream,a.out,device,a.resume,budget_hours=a.budget_hours)
    if result['status']!='COMPLETED_VERIFIED':raise SystemExit(3)


if __name__=='__main__':main()
