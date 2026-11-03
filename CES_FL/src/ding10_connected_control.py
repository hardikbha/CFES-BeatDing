#!/usr/bin/env python3
"""Seed-17 ten-epoch connected-gradient Ding controlled adaptation.

This is a separately named controlled variant of the archived Ding SFD+GDA
recipe. It reuses the seed-17 classification checkpoint and TRAIN-derived
10/512 channel mask, freezes the encoder and BN buffers, and connects the
Sinkhorn fairness term to the main prediction logits. It is not a reproduction
of the paper's reported 50-epoch protocol.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn.utils import prune
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

VERSION = 'DING10_CONNECTED_CONTROL_v1'
GROUPS = ('Male-Asian','Male-White','Male-Black','Male-Others',
          'Female-Asian','Female-White','Female-Black','Female-Others')
GROUP_ID = {g:i for i,g in enumerate(GROUPS)}
CLASS_NAMES = ('real','fake')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,p)


def atomic_torch(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.tmp')
    torch.save(value,tmp);os.replace(tmp,p)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def read_rows(path,require_native_groups=True):
    p=Path(path)
    with p.open(newline='') as f:
        reader=csv.DictReader(f)
        required={'image','label','gender_race','identity_ids','source_ids'}
        if not required.issubset(reader.fieldnames or []):raise ValueError(f'Missing columns {required-set(reader.fieldnames or [])}')
        rows=list(reader)
    seen=set()
    for row in rows:
        image=Path(row['image']).resolve();row['image']=str(image)
        if row['label'] not in ('0','1') or not image.is_file():raise ValueError(f'Invalid image/label: {image}')
        row['label']=int(row['label'])
        if image in seen:raise ValueError('Duplicate image row')
        seen.add(image)
        group=row['gender_race']
        if require_native_groups and group not in GROUP_ID:raise ValueError(f'Unknown native group {group}')
        row['group_id']=GROUP_ID.get(group,-1)
        for key in ('identity_ids','source_ids'):
            if not row[key].strip():raise ValueError(f'Empty {key}')
    if {r['label'] for r in rows}!={0,1}:raise ValueError('Both classes required')
    return rows


def split_audit(train,val,overlap_path):
    if {r['image'] for r in train}&{r['image'] for r in val}:raise ValueError('Image path overlap')
    declared=json.loads(Path(overlap_path).read_text()).get('declared_overlap_values',{})
    observed={}
    for key in ('identity_ids','source_ids'):
        a={v for r in train for v in r[key].split(';') if v}
        b={v for r in val for v in r[key].split(';') if v}
        overlap=sorted(a&b)
        if overlap!=declared.get(f'{key}:train:val',[]):raise ValueError(f'Unrecorded {key} train/val overlap')
        observed[key]=len(overlap)
    return {'recorded_proxy_overlaps':observed,'person_identity_separation_verified':False}


def check_config(c,bind=True):
    expected={'format','seed','epochs','main_batch','aux_per_class','updates_per_epoch','workers','size','eval_batch_size',
        'lr','momentum','weight_decay','lambda_fair','sinkhorn_blur','target_fpr','min_group_support',
        'selection','train','val','overlap','input_hashes','ding_code_root','base_checkpoint','sfd_channels',
        'sfd_done','sfd_log','native_gda','upstream_hashes','source_sha256','dataset_id','bn_policy',
        'selection_thresholds','main_sampling','group_sampling','normalization','ding_realtrain_csv','ding_faketrain_csv'}
    if set(c)!=expected:raise ValueError(f'Config schema mismatch: {expected-set(c)} / {set(c)-expected}')
    if c['format']!=VERSION or c['seed']!=17 or c['epochs']!=10:raise ValueError('Wrong protocol ID, seed or ten-epoch budget')
    if (c['main_batch'],c['aux_per_class'],c['updates_per_epoch'],c['workers'],c['size'],c['eval_batch_size'])!=(32,32,2380,4,224,64):
        raise ValueError('Fixed Ding-10 controlled settings changed')
    if c['selection']!='earliest_strict_max_validation_auc' or c['selection_thresholds']!=['fixed_probability_0_5','validation_real_empirical_10pct']:
        raise ValueError('Selection/threshold protocol changed')
    if c['bn_policy']!='frozen_running_stats' or c['normalization']!='RGB bilinear direct 224x224, float [0,1], (x-.5)/.5':
        raise ValueError('Preprocessing or BN policy changed')
    if c['main_sampling']!='shuffle all TRAIN originals once per epoch, no replacement':
        raise ValueError('Original exposure changed')
    if c['group_sampling']!='uniform group; 32 TRAIN real + 32 TRAIN fake with replacement':
        raise ValueError('Group target sampling changed')
    if any(not math.isfinite(c[k]) or c[k]<=0 for k in ('lr','momentum','weight_decay','lambda_fair','sinkhorn_blur')):
        raise ValueError('Invalid optimizer/fairness settings')
    if c['lr']!=.001 or c['momentum']!=.9 or c['weight_decay']!=.005 or c['lambda_fair']!=.005 or c['sinkhorn_blur']!=5e-4:
        raise ValueError('Ding controlled hyperparameters changed')
    bound_paths=('train','val','overlap','base_checkpoint','sfd_channels','sfd_done','sfd_log','ding_realtrain_csv','ding_faketrain_csv')
    if set(c['input_hashes'])!=set(bound_paths):raise ValueError('Input hash inventory changed')
    for key in bound_paths:
        if sha(c[key])!=c['input_hashes'][key]:raise ValueError(f'{key} hash mismatch')
    if Path(c['native_gda']).resolve()!=Path(c['ding_code_root'],'GDA.py').resolve():raise ValueError('Native GDA reference path changed')
    for key,path in (('gda',Path(c['ding_code_root'])/'GDA.py'),
                     ('xception',Path(c['ding_code_root'])/'utils/xception.py'),
                     ('detector',Path(c['ding_code_root'])/'detectors/xception_detector.py')):
        if sha(path)!=c['upstream_hashes'][key]:raise ValueError(f'Pinned Ding source changed: {key}')
    channels=[int(x) for x in Path(c['sfd_channels']).read_text().split()]
    if len(channels)!=10 or len(set(channels))!=10 or any(x<0 or x>=512 for x in channels):
        raise ValueError('SFD file must list ten unique indices in [0,511]')
    log=Path(c['sfd_log']).read_text(errors='replace')
    if c['base_checkpoint'] not in log or '--prc 2.0' not in log or c['sfd_channels'] not in log:
        raise ValueError('SFD run log does not bind base checkpoint, 2% and mask file')
    if bind and c['source_sha256']!=sha(__file__):raise ValueError('Ding adapter source hash mismatch')
    train=read_rows(c['train']);val=read_rows(c['val'])
    audit=split_audit(train,val,c['overlap'])
    cc=Counter((r['label'],r['group_id']) for r in train)
    missing=[(y,g) for y in (0,1) for g in range(8) if cc[(y,g)]==0]
    if missing:raise ValueError(f'Missing TRAIN class/group cells: {missing}')
    return c,train,val,audit,{str(g):{'name':GROUPS[g],'real':cc[(0,g)],'fake':cc[(1,g)]} for g in range(8)},channels


def compare_native_ding_membership(c,rows):
    mapped={r['image']:(r['label'],r['gender_race']) for r in rows}
    raw={}
    for path in (c['ding_realtrain_csv'],c['ding_faketrain_csv']):
        with Path(path).open(newline='') as f:
            reader=csv.DictReader(f)
            if not {'img_path','label','intersec_label'}.issubset(reader.fieldnames or []):
                raise ValueError('Native Ding train CSV schema changed')
            for row in reader:
                image=str(Path(row['img_path']).resolve())
                group_code=int(row['intersec_label'])
                if not 0<=group_code<8:raise ValueError('Native Ding group code outside 0..7')
                if image in raw:raise ValueError('Duplicate native Ding original path')
                raw[image]=(int(row['label']),GROUPS[group_code])
    if raw!=mapped:
        missing=set(mapped)-set(raw);extra=set(raw)-set(mapped)
        mismatch=next(((p,mapped[p],raw[p]) for p in mapped.keys()&raw.keys() if mapped[p]!=raw[p]),None)
        raise ValueError(f'Native Ding membership/group mismatch: missing={len(missing)} extra={len(extra)} mismatch={mismatch}')
    return {'status':'PASS','rows':len(raw),'path_label_group_exact':True}


class MainImages(Dataset):
    def __init__(self,rows,size=224):self.rows,self.size=rows,size
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row['image']) as im:
            im=im.convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR)
            a=np.asarray(im,dtype=np.float32).copy()/255.
        x=(torch.from_numpy(a).permute(2,0,1)-.5)/.5
        return x,row['label'],row['group_id'],index


class MainBatchSampler(Sampler):
    def __init__(self,n,batch,seed,epoch):self.n,self.batch,self.seed,self.epoch=n,batch,seed,epoch
    def __iter__(self):
        ix=np.random.default_rng(self.seed+self.epoch).permutation(self.n).tolist()
        return iter([ix[i:i+self.batch] for i in range(0,self.n,self.batch)])
    def __len__(self):return (self.n+self.batch-1)//self.batch


class AuxTargets(Dataset):
    def __init__(self,rows,seed,epoch,steps,size=224,per_class=32):
        self.rows,self.seed,self.epoch,self.steps,self.size,self.per_class=rows,seed,epoch,steps,size,per_class
        self.cells=defaultdict(lambda:defaultdict(list))
        for i,r in enumerate(rows):self.cells[r['group_id']][r['label']].append(i)
        if any(not self.cells[g][y] for g in range(8) for y in (0,1)):
            raise ValueError('Auxiliary group/class cell is empty')
    def __len__(self):return self.steps
    def _load(self,row):
        with Image.open(row['image']) as im:
            im=im.convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR)
            a=np.asarray(im,dtype=np.float32).copy()/255.
        return (torch.from_numpy(a).permute(2,0,1)-.5)/.5
    def __getitem__(self,step):
        rng=np.random.default_rng(self.seed+1000003*self.epoch+step)
        group=int(rng.integers(0,8));indices=[];labels=[]
        for label in (0,1):
            ids=self.cells[group][label]
            indices.extend(rng.choice(ids,size=self.per_class,replace=True).tolist())
            labels.extend([label]*self.per_class)
        return torch.stack([self._load(self.rows[i]) for i in indices]),torch.tensor(labels),group


class DingXception(nn.Module):
    def __init__(self,base_checkpoint,channels,code_root):
        super().__init__();sys.path.insert(0,str(code_root))
        from utils.xception import Xception
        self.backbone=Xception({'mode':'adjust_channel','num_classes':2,'inc':3,'dropout':False})
        raw=torch.load(base_checkpoint,map_location='cpu',weights_only=True)
        self.load_state_dict(raw,strict=True)
        layer=self.backbone.adjust_channel[0]
        mask=torch.ones_like(layer.weight)
        for channel in channels:mask[channel]=0
        prune.custom_from_mask(layer,name='weight',mask=mask)
        for p in self.parameters():p.requires_grad_(False)
        for p in self.backbone.last_linear.parameters():p.requires_grad_(True)
        self.freeze_bn()
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
    def train(self,mode=True):
        super().train(mode);self.freeze_bn();return self
    def forward(self,x):return self.backbone(x)[0]
    def trainable_classifier(self):return list(self.backbone.last_linear.parameters())


def objective(main_logits,labels,reference_logits,sinkhorn,lambda_fair=.005):
    cls=F.cross_entropy(main_logits,labels)
    # Connected-gradient controlled variant: main distribution is optimized;
    # the balanced group sample is a fixed target as in native group sampling.
    fair=sinkhorn(main_logits,reference_logits.detach())
    return {'total':cls+lambda_fair*fair,'classification':cls,'fairness':fair}


def calibrate(real_prob,target=.1):
    z=np.asarray(real_prob,dtype=np.float64)
    if z.ndim!=1 or len(z)==0 or not np.isfinite(z).all():raise ValueError('Bad calibration scores')
    k=int(math.floor(target*len(z)))
    threshold=float(np.nextafter(np.sort(z)[::-1][k],np.inf))
    fp=int((z>=threshold).sum())
    return {'probability_threshold':threshold,'n_validation_real':int(len(z)),'allowed_fp':k,'achieved_fp':fp,
            'achieved_fpr':fp/len(z),'target_fpr':target,'test_fpr_guaranteed':False}


def predict(model,rows,device,batch_size=64,workers=4,size=224):
    model.eval();values=[]
    loader=DataLoader(MainImages(rows,size),batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=True)
    with torch.inference_mode():
        for x,y,g,i in loader:
            logits=model(x.to(device,non_blocking=True));values.extend(torch.softmax(logits.float(),1)[:,1].cpu().tolist())
    return np.asarray(values,dtype=np.float64)


def group_metrics(rows,prob,threshold,min_support=20):
    from sklearn.metrics import roc_auc_score
    y=np.asarray([r['label'] for r in rows]);pred=prob>=threshold
    def calc(ix):
        yy=y[ix];pp=pred[ix];zz=prob[ix];real=yy==0;fake=yy==1
        return {'n':int(len(yy)),'n_real':int(real.sum()),'n_fake':int(fake.sum()),
            'auc':float(roc_auc_score(yy,zz)) if real.any() and fake.any() else None,
            'fpr':float(pp[real].mean()) if real.any() else None,'fnr':float((~pp[fake]).mean()) if fake.any() else None,
            'positive_rate':float(pp.mean()) if len(pp) else None}
    overall=calc(np.ones(len(y),dtype=bool));axes={}
    for axis in ('gender','race','gender_race'):
        names=sorted({r[axis] for r in rows if r.get(axis) not in ('','unknown')})
        groups={n:calc(np.asarray([r.get(axis)==n for r in rows])) for n in names}
        eligible=[n for n,g in groups.items() if min(g['n_real'],g['n_fake'])>=min_support]
        valid=[groups[n] for n in eligible]
        fpr_values=[g['fpr'] for g in valid if g['fpr'] is not None]
        fnr_values=[g['fnr'] for g in valid if g['fnr'] is not None]
        pr_values=[g['positive_rate'] for g in valid if g['positive_rate'] is not None]
        axes[axis]={'groups':groups,'eligible_groups':eligible,'min_support_per_class':min_support,
            'worst_group_auc':min(g['auc'] for g in valid) if valid and all(g['auc'] is not None for g in valid) else None,
            'F_FPR':sum(abs(g['fpr']-overall['fpr']) for g in valid) if len(fpr_values)>=2 else None,
            'F_DP':max(pr_values)-min(pr_values) if len(pr_values)>=2 else None,
            'FNR_gap':max(fnr_values)-min(fnr_values) if len(fnr_values)>=2 else None}
    return {'units':'fractions','overall':overall,'axes':axes,'threshold_probability':threshold}


def validation_report(model,rows,device,c):
    prob=predict(model,rows,device,c['eval_batch_size'],c['workers'],c['size'])
    labels=np.asarray([r['label'] for r in rows]);cal=calibrate(prob[labels==0],c['target_fpr'])
    return {'probabilities':prob,'calibration':cal,'calibrated':group_metrics(rows,prob,cal['probability_threshold'],c['min_group_support']),
            'fixed_probability_0_5':group_metrics(rows,prob,.5,c['min_group_support'])}


def dry_step(c,rows,channels,device,synthetic=False):
    train=rows
    model=DingXception(c['base_checkpoint'],channels,c['ding_code_root']).to(device)
    model.train();classifier=model.trainable_classifier()
    bn_before={n:b.detach().clone() for n,b in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
    mask_before=model.backbone.adjust_channel[0].weight_mask.detach().clone()
    opt=torch.optim.SGD(classifier,lr=c['lr'],momentum=c['momentum'],weight_decay=c['weight_decay'])
    rng=np.random.default_rng(c['seed']+404)
    if synthetic:
        main_x=torch.randn(4,3,c['size'],c['size'],device=device)
        labels=torch.tensor([0,0,1,1],device=device)
        ref_x=torch.randn(8,3,c['size'],c['size'],device=device)
    else:
        class_rows={y:[i for i,r in enumerate(train) if r['label']==y] for y in (0,1)}
        main_ix=[int(rng.choice(class_rows[y])) for y in (0,1) for _ in range(16)]
        main_ds=MainImages(train,c['size'])
        main_items=[main_ds[i] for i in main_ix]
        main_x=torch.stack([x for x,y,g,i in main_items]).to(device)
        labels=torch.tensor([y for x,y,g,i in main_items],device=device,dtype=torch.long)
        group=int(rng.integers(0,8));cells={y:[i for i,r in enumerate(train) if r['group_id']==group and r['label']==y] for y in (0,1)}
        if any(not cells[y] for y in (0,1)):raise ValueError('Auxiliary dry-run group/class empty')
        aux_ix=[int(rng.choice(cells[y])) for y in (0,1) for _ in range(32)]
        aux_ds=MainImages(train,c['size']);ref_x=torch.stack([aux_ds[i][0] for i in aux_ix]).to(device)
    if labels.min()!=0 or labels.max()!=1:raise ValueError('Dry-run batch must include both classes')
    # Reference Sinkhorn has no gradient; connected gradient comes only from main logits.
    sinkhorn=load_samples_loss(c)
    with torch.no_grad():ref_logits=model(ref_x)
    main_logits=model(main_x)
    losses=objective(main_logits,labels,ref_logits,sinkhorn,c['lambda_fair'])
    fair_grad=torch.autograd.grad(losses['fairness'],main_logits,retain_graph=True)[0]
    grad_bce=torch.autograd.grad(losses['classification'],main_logits,retain_graph=True)[0]
    grad_connected=torch.autograd.grad(losses['total'],main_logits,retain_graph=True)[0]
    update_delta=c['lr']*float((grad_connected-grad_bce).float().norm())
    if not math.isfinite(update_delta) or update_delta<=0:raise RuntimeError('Fairness coefficient does not change the SGD update')
    opt.zero_grad(set_to_none=True);losses['total'].backward()
    grad=model.backbone.last_linear.weight.grad
    if grad is None or not torch.isfinite(grad).all() or torch.count_nonzero(grad)==0:
        raise RuntimeError('Fairness/classification backward produced no finite classifier gradient')
    if not torch.isfinite(fair_grad).all() or not torch.count_nonzero(fair_grad):raise RuntimeError('Fairness gradient is detached')
    p_before=[p.detach().clone() for p in classifier]
    opt.step()
    if not any(not torch.equal(a,p.detach()) for a,p in zip(p_before,classifier)):
        raise RuntimeError('Classifier optimizer update did not occur')
    buffers=dict(model.named_buffers())
    if any(not torch.equal(v,buffers[k]) for k,v in bn_before.items()):raise RuntimeError('BN buffer changed')
    if not torch.equal(mask_before,model.backbone.adjust_channel[0].weight_mask):raise RuntimeError('SFD mask changed')
    if any(m.training for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)):
        raise RuntimeError('BatchNorm module entered training mode')
    if any(p.grad is not None for n,p in model.named_parameters() if n!='backbone.last_linear.weight' and n!='backbone.last_linear.bias'):
        raise RuntimeError('Frozen non-classifier parameter received a gradient')
    return {'status':'PASS','source_sha256':sha(__file__),'device':str(device),'synthetic_fixture':synthetic,
        'fairness_gradient_norm':float(fair_grad.float().norm()),'classifier_gradient_norm':float(grad.float().norm()),
        'lambda_update_delta_norm':update_delta,
        'losses':{k:float(v.detach()) for k,v in losses.items()},'mask_channels':channels,
        'classifier_parameters':sum(p.numel() for p in classifier),'optimizer_steps_in_discarded_model':1,
        'persistent_optimizer_updates':0,
        'real_input_batch_size':0 if synthetic else int(len(labels)),'test_read':False,
        'bn_frozen':True,'only_last_linear_trainable':True,'gradient_connected':True}


def load_samples_loss(c):
    from geomloss import SamplesLoss
    return SamplesLoss(loss='sinkhorn',p=2,blur=c['sinkhorn_blur'])


def smoke(c,out,device='cpu'):
    out=Path(out)
    if out.exists():raise ValueError(f'Preserve existing smoke path: {out}')
    out.mkdir(parents=True)
    c,train,val,audit,coverage,channels=preflight(c)
    synthetic=dry_step(c,train,channels,torch.device(device),synthetic=True)
    sinkhorn=load_samples_loss(c)
    reference=torch.randn(8,2,device=device)
    labels=torch.tensor([0,1,0,1],device=device)
    initial=torch.tensor([[.2,-.1],[-.3,.4],[.6,-.2],[-.1,.3]],device=device)
    momentum=[]
    for lam in (0.,c['lambda_fair']):
        logits=nn.Parameter(initial.clone());opt=torch.optim.SGD([logits],lr=c['lr'],momentum=c['momentum'],weight_decay=c['weight_decay'])
        losses=objective(logits,labels,reference,sinkhorn,lam);opt.zero_grad(set_to_none=True);losses['total'].backward();opt.step()
        momentum.append(opt.state[logits]['momentum_buffer'].detach().clone())
    lambda_delta=float((momentum[1]-momentum[0]).norm())
    if not math.isfinite(lambda_delta) or lambda_delta<=0:raise RuntimeError('Changing fairness weight did not change SGD update state')
    tied=calibrate([.9,.9,.3,.1],.25)
    if tied['achieved_fpr']>.25:raise RuntimeError('Finite-sample tied-score calibration exceeded target')
    synthetic['lambda_update_delta_norm']=lambda_delta
    synthetic['tie_calibration_fixture']=tied
    model=DingXception(c['base_checkpoint'],channels,c['ding_code_root']).to(device).eval()
    expected_mask=10*model.backbone.adjust_channel[0].weight_mask.shape[1]*model.backbone.adjust_channel[0].weight_mask.shape[2]*model.backbone.adjust_channel[0].weight_mask.shape[3]
    if int((model.backbone.adjust_channel[0].weight_mask==0).sum())!=expected_mask:
        raise RuntimeError('Fixed SFD mask element count differs from ten output channels')
    sample=torch.randn(2,3,c['size'],c['size'],device=device)
    with torch.no_grad():before=model(sample)
    with tempfile.TemporaryDirectory(prefix='ding10_head_reload_') as tmp:
        path=Path(tmp)/'head.pt';torch.save({'classifier_state':model.backbone.last_linear.state_dict(),'mask_channels':channels},path)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        restored=DingXception(c['base_checkpoint'],channels,c['ding_code_root']).to(device).eval()
        restored.backbone.last_linear.load_state_dict(saved['classifier_state'],strict=True)
        with torch.no_grad():after=restored(sample)
        if not torch.equal(before,after):raise RuntimeError('Ding head checkpoint reload mismatch')
    synthetic['strict_head_reload']=True
    synthetic['sfd_zeroed_weight_elements']=expected_mask
    write_json(out/'SMOKE.json',synthetic)
    model=DingXception(c['base_checkpoint'],channels,c['ding_code_root']).to(device).eval()
    sample=torch.randn(2,3,c['size'],c['size'],device=device)
    with torch.no_grad():baseline=model(sample)
    head_state={k:v.detach().cpu().clone() for k,v in model.backbone.last_linear.state_dict().items()}
    restored=DingXception(c['base_checkpoint'],channels,c['ding_code_root']).to(device).eval()
    restored.backbone.last_linear.load_state_dict(head_state,strict=True)
    with torch.no_grad():reloaded=restored(sample)
    if not torch.equal(baseline,reloaded):raise RuntimeError('Ding classifier checkpoint reload differs')
    mask=model.backbone.adjust_channel[0].weight_mask
    if int((mask==0).sum())!=10*mask.shape[1]*mask.shape[2]*mask.shape[3]:
        raise RuntimeError('SFD mask does not zero exactly ten conv output channels')
    write_json(out/'MODEL_RELOAD_AND_SFD_MASK.json',{'status':'PASS','strict_reload':True,
        'mask_channels':channels,'masked_weight_elements':int((mask==0).sum()),'test_read':False})
    write_json(out/'PREFLIGHT.json',{'status':'PASS','train_count':len(train),'val_count':len(val),
        'split_audit':audit,'group_counts':coverage,'base_sha256':sha(c['base_checkpoint']),
        'sfd_sha256':sha(c['sfd_channels']),'test_read':False})
    return {'smoke':synthetic,'preflight':coverage}


def preflight(c):
    c,train,val,audit,coverage,channels=preflight_impl(c)
    return c,train,val,audit,coverage,channels


def preflight_impl(c):
    check_config(c)
    train=read_rows(c['train']);val=read_rows(c['val'])
    audit=split_audit(train,val,c['overlap'])
    audit['native_Ding_membership']=compare_native_ding_membership(c,train)
    counts=Counter((r['label'],r['group_id']) for r in train)
    coverage={GROUPS[g]:{'real':counts[(0,g)],'fake':counts[(1,g)]} for g in range(8)}
    return c,train,val,audit,coverage,[int(x) for x in Path(c['sfd_channels']).read_text().split()]


def _close_tree(a,b,tol=1e-8):
    if isinstance(a,dict):
        return set(a)==set(b) and all(_close_tree(a[k],b[k],tol) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(_close_tree(x,y,tol) for x,y in zip(a,b))
    if isinstance(a,(float,int)) and not isinstance(a,bool) and isinstance(b,(float,int)):
        return abs(float(a)-float(b))<=tol
    return a==b


def verify_selected(out,c,device,channels):
    out=Path(out);ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    if ck.get('format')!=VERSION or ck.get('source_sha256')!=sha(__file__) or ck.get('config')!=c:
        raise ValueError('Selected Ding checkpoint provenance mismatch')
    model=DingXception(c['base_checkpoint'],channels,c['ding_code_root']).to(device)
    model.backbone.last_linear.load_state_dict(ck['classifier_state'],strict=True)
    _,_,val,_,_,_=preflight(c)
    report=validation_report(model,val,device,c)
    replay={'calibrated':report['calibrated'],'fixed_probability_0_5':report['fixed_probability_0_5']}
    expected={'calibrated':ck['validation'],'fixed_probability_0_5':ck['fixed_probability_0_5']}
    if not _close_tree(replay,expected) or not _close_tree(report['calibration'],ck['calibration']):
        raise RuntimeError('Selected validation replay mismatch')
    receipt={'status':'PASS','arm':'Ding SFD+GDA, connected-gradient, 10-epoch controlled adaptation',
        'checkpoint_sha256':sha(out/'best.pt'),'source_sha256':sha(__file__),'config_sha256':digest(c),
        'upstream_hashes':c['upstream_hashes'],'completed_epochs':c['epochs'],'selected_epoch':ck['best_epoch'],
        'selected_validation_auc':ck['best_auc'],'validation_replayed':True,'test_read':False,
        'fairness_improvement_verified':False,'validation':ck['validation']}
    write_json(out/'RUN_VERIFIED.json',receipt)
    return receipt


def train_run(c,out,device,resume=False):
    c,train,val,split,coverage,channels=preflight(c)
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if device.type=='cuda' and torch.cuda.device_count()!=1:raise RuntimeError('One direct physical GPU must be bound')
    torch.set_num_threads(4);torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=DingXception(c['base_checkpoint'],channels,c['ding_code_root']).to(device)
    classifier=model.trainable_classifier()
    optimizer=torch.optim.SGD(classifier,lr=c['lr'],momentum=c['momentum'],weight_decay=c['weight_decay'])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=60,gamma=.9)
    sinkhorn=load_samples_loss(c)
    out=Path(out).resolve();start_epoch=0;best_auc=-math.inf;best_epoch=None;history=[]
    if resume:
        if (out/'DONE').exists():raise ValueError('Completed run is immutable')
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if state.get('format')!=VERSION or state.get('source_sha256')!=sha(__file__) or state.get('config')!=c:
            raise ValueError('Resume source/config mismatch')
        history=[json.loads(line) for line in (out/'history.jsonl').read_text().splitlines()]
        if [h['epoch'] for h in history]!=list(range(1,state['epoch']+1)):raise ValueError('History/checkpoint mismatch')
        model.backbone.last_linear.load_state_dict(state['classifier_state'],strict=True)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start_epoch=state['epoch'];best_auc=state['best_auc'];best_epoch=state['best_epoch']
        for v in optimizer.state.values():
            for k,x in v.items():
                if torch.is_tensor(x):v[k]=x.to(device)
    else:
        if out.exists():raise ValueError(f'Preserve existing output directory {out}')
        out.mkdir(parents=True)
        write_json(out/'config.json',c)
        write_json(out/'provenance.json',{'source_sha256':sha(__file__),'upstream_hashes':c['upstream_hashes'],
            'base_checkpoint_sha256':sha(c['base_checkpoint']),'sfd_channels_sha256':sha(c['sfd_channels']),
            'sfd_channels':channels,'train_sha256':sha(c['train']),'val_sha256':sha(c['val']),
            'split_audit':split,'group_counts':coverage,'trainable':'last_linear only (1,026 params)',
            'bn_buffers':'frozen','optimizer':'SGD lr=.001 momentum=.9 weight_decay=.005',
            'fairness':'connected SamplesLoss(sinkhorn,p=2,blur=5e-4), group target detached, lambda=.005',
            'main_batch_images':32,'balanced_group_reference_images':64,'test_read':False,
            'paper_protocol_equivalent':False,'controlled_adaptation':True})
    dataset=MainImages(train,c['size']);steps=(len(train)+c['main_batch']-1)//c['main_batch']
    if steps!=c['updates_per_epoch']:raise ValueError(f'Expected {c["updates_per_epoch"]} steps, got {steps}')
    aux_counts=Counter();group_main=Counter();start=time.monotonic()
    for epoch in range(start_epoch,c['epochs']):
        model.train();batch_sampler=MainBatchSampler(len(train),c['main_batch'],c['seed'],epoch)
        main_loader=DataLoader(dataset,batch_sampler=batch_sampler,num_workers=c['workers'],pin_memory=True,
            persistent_workers=False)
        aux_ds=AuxTargets(train,c['seed'],epoch,steps,c['size'],c['aux_per_class'])
        aux_loader=DataLoader(aux_ds,batch_size=None,num_workers=2,pin_memory=True,prefetch_factor=1)
        sums=Counter();seen=Counter();group_sampling=Counter();step_count=0
        for step,(main_batch,aux_batch) in enumerate(zip(main_loader,aux_loader)):
            x,y,g,indices=main_batch
            aux_x,aux_y,aux_g=aux_batch
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True).long()
            aux_x=aux_x.to(device,non_blocking=True)
            model.train()
            logits=model(x)
            with torch.no_grad():ref_logits=model(aux_x)
            losses=objective(logits,y,ref_logits,sinkhorn,c['lambda_fair'])
            if not torch.isfinite(losses['total']):raise RuntimeError('Nonfinite Ding-10 loss')
            optimizer.zero_grad(set_to_none=True);losses['total'].backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in classifier):
                raise RuntimeError('Missing or nonfinite last_linear gradient')
            if any(p.grad is not None for n,p in model.named_parameters() if 'last_linear' not in n):
                raise RuntimeError('Frozen Ding parameter received gradient')
            optimizer.step();step_count+=1
            seen.update(indices.tolist());group_sampling[int(aux_g)]+=1
            group_main.update(int(v) for v in g.tolist())
            aux_counts[f'{int(aux_g)}:real']+=int((aux_y==0).sum())
            aux_counts[f'{int(aux_g)}:fake']+=int((aux_y==1).sum())
            for k,v in losses.items():sums[k]+=float(v.detach())
            if (step+1)%200==0:print(json.dumps({'stage':'Ding10','epoch':epoch+1,'step':step+1,'steps':steps}),flush=True)
        if step_count!=c['updates_per_epoch'] or len(seen)!=len(train) or any(n!=1 for n in seen.values()):
            raise RuntimeError('TRAIN exposure/update count mismatch')
        scheduler.step()
        result=validation_report(model,val,device,c)
        auc=result['calibrated']['overall']['auc'];selected=auc>best_auc
        if selected:best_auc=auc;best_epoch=epoch+1
        record={'epoch':epoch+1,'steps':step_count,'updates_expected':c['updates_per_epoch'],
            'validation':result['calibrated'],'fixed_probability_0_5':result['fixed_probability_0_5'],
            'calibration':result['calibration'],'selection':'earliest_strict_max_validation_auc',
            'selected_this_epoch':selected,'losses':{k:float(v)/step_count for k,v in sums.items()},
            'unique_originals':len(seen),'originals_once_each':True,
            'main_group_counts':dict(group_main),'aux_group_updates':dict(group_sampling),
            'balanced_aux_examples':dict(aux_counts),'elapsed_seconds':time.monotonic()-start}
        history.append(record)
        ck={'format':VERSION,'source_sha256':sha(__file__),'upstream_hashes':c['upstream_hashes'],
            'config':c,'epoch':epoch+1,'best_auc':best_auc,'best_epoch':best_epoch,
            'classifier_state':{k:v.detach().cpu().clone() for k,v in model.backbone.last_linear.state_dict().items()},
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'calibration':result['calibration'],'validation':result['calibrated'],'history':history,
            'sfd_channels':channels,'test_read':False}
        if selected:atomic_torch(out/'best.pt',ck)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        atomic_torch(out/'last.pt',ck)
        print(json.dumps({'stage':'Ding10','epoch':epoch+1,'val_auc':auc,'selected':selected}),flush=True)
    receipt=verify_selected(out,c,device,channels)
    write_json(out/'summary.json',{'status':'COMPLETED_VERIFIED','epochs':c['epochs'],
        'selected_epoch':best_epoch,'selected_val_auc':best_auc,'test_read':False,'receipt':receipt})
    (out/'DONE').write_text('Ding connected-gradient controlled adaptation completed; test not read.\n')
    return receipt


def evaluate(checkpoint,manifest,manifest_hash,out,device):
    run_dir=Path(checkpoint).resolve().parent
    if not (run_dir/'DONE').is_file() or not (run_dir/'RUN_VERIFIED.json').is_file():
        raise ValueError('TEST evaluation requires a completed, validation-replayed run')
    receipt=json.loads((run_dir/'RUN_VERIFIED.json').read_text())
    if receipt.get('status')!='PASS' or receipt.get('test_read') is not False:
        raise ValueError('Invalid or TEST-exposed run receipt')
    if sha(manifest)!=manifest_hash:raise ValueError('Test manifest hash mismatch')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck.get('format')!=VERSION or ck.get('source_sha256')!=sha(__file__):raise ValueError('Ding checkpoint source mismatch')
    c=ck['config'];_,train,val,audit,coverage,channels=preflight(c)
    p=Path(manifest).resolve()
    test=read_rows(p,require_native_groups=False)
    model=DingXception(c['base_checkpoint'],channels,c['ding_code_root']).to(device)
    model.backbone.last_linear.load_state_dict(ck['classifier_state'],strict=True);model.eval()
    # Verify the declared fit:test proxy exception before scoring.
    declared=json.loads(Path(c['overlap']).read_text()).get('declared_overlap_values',{})
    for k in ('identity_ids','source_ids'):
        a={v for r in train for v in r[k].split(';') if v};b={v for r in test for v in r[k].split(';') if v}
        cset={v for r in val for v in r[k].split(';') if v}
        observed=sorted(a&b);expected=sorted(declared.get(f'{k}:train:test',[]))
        if observed!=expected:raise ValueError(f'Undeclared fit:test {k} overlap')
        val_observed=sorted(cset&b)
        if val_observed!=sorted(declared.get(f'{k}:val:test',[])):raise ValueError(f'Undeclared val:test {k} overlap')
    prob=predict(model,test,device,c['eval_batch_size'],c['workers'],c['size'])
    threshold=ck['calibration']['probability_threshold']
    report={'checkpoint_sha256':sha(checkpoint),'manifest_sha256':manifest_hash,'source_sha256':sha(__file__),
        'threshold_source':'selected FF++ validation real images only','calibration':ck['calibration'],
        'calibrated':group_metrics(test,prob,threshold,c['min_group_support']),
        'fixed_probability_0_5':group_metrics(test,prob,.5,c['min_group_support']),
        'test_read_during_training':False,'target_domain_calibration':False,
        'split_audit':{'recorded_proxy_overlap_counts':{'identity_ids_train_test':len(declared.get('identity_ids:train:test',[])),
            'identity_ids_val_test':len(declared.get('identity_ids:val:test',[])),
            'source_ids_train_test':len(declared.get('source_ids:train:test',[])),
            'source_ids_val_test':len(declared.get('source_ids:val:test',[]))},
            'person_identity_ground_truth_verified':False}}
    out=Path(out)
    if out.exists():raise ValueError('Preserve existing test evaluation output')
    out.mkdir(parents=True)
    write_json(out/'metrics.json',report)
    write_json(out/'scores_PRIVATE.json',{'rows':test,'prob_fake':prob.tolist(),'threshold':threshold,
        'manifest_sha256':manifest_hash,'checkpoint_sha256':sha(checkpoint),'test_read':True})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('preflight');p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('smoke');p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p=sub.add_parser('dry-run');p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p=sub.add_parser('train');p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--resume',action='store_true')
    p=sub.add_parser('evaluate');p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--out',required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args();c=json.loads(Path(args.config).read_text())
    if args.cmd=='preflight':
        _,train,val,audit,coverage,channels=preflight(c)
        out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
        write_json(out/'PREFLIGHT.json',{'status':'PASS','train':len(train),'val':len(val),'audit':audit,'coverage':coverage,'channels':channels,'test_read':False})
    elif args.cmd=='smoke':smoke(c,args.out,args.device)
    elif args.cmd=='dry-run':
        _,train,val,audit,coverage,channels=preflight(c)
        result=dry_step(c,train,channels,torch.device(args.device),synthetic=False)
        out=Path(args.out);out.mkdir(parents=True,exist_ok=False);write_json(out/'DRY_RUN.json',result)
    elif args.cmd=='train':train_run(c,args.out,args.device,args.resume)
    else:evaluate(args.checkpoint,args.manifest,args.manifest_sha256,args.out,args.device)


if __name__=='__main__':main()
