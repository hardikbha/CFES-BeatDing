#!/usr/bin/env python3
"""Frozen-encoder source-tail experiment. TRAIN/SELECT only; cached features.

Stages: prepare (one encoder pass), train (small head, CPU supported), export.
No TEST/CAL command, model download, automatic GPU allocation or dataset edits.
"""
import argparse,csv,hashlib,importlib.util,json,math,signal,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

HERE=Path(__file__).resolve().parent
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
core=load(HERE/'teacher_core.py','source_tail_teacher_core')
objective=load(HERE.parents[1]/'src/source_tail_objective.py','source_tail_loss')
def hashes():return {p.name:core.sha(p) for p in (Path(__file__),Path(core.__file__),Path(objective.__file__),HERE/'config.json')}
def config():
    c=json.loads((HERE/'config.json').read_text())
    if core.sha(core.__file__)!=c['teacher_core_sha256']:raise ValueError('Teacher implementation changed')
    return c

class Images(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return core.normalize(core.load_pixels(self.rows[i],224)),i

@torch.inference_mode()
def prepare(c,teacher_run,source_csv,source_provenance,out,device):
    teacher_run=Path(teacher_run);checkpoint=teacher_run/'best.pt'
    if not (teacher_run/'DONE').is_file() or core.sha(checkpoint)!=c['teacher_checkpoint_sha256']:
        raise ValueError('Exact completed L_A teacher required')
    v=json.loads((teacher_run/'RUN_VERIFIED.json').read_text())
    if v['status']!='COMPLETED_VERIFIED' or v['arm']!='A' or v.get('synthetic_fixture') or v['selected_checkpoint_sha256']!=c['teacher_checkpoint_sha256']:
        raise ValueError('Not a verified original-image baseline')
    model,ck=core.load_trained(checkpoint,torch.device('cpu'))
    if ck['config']['head_type']!='linear':raise ValueError('Complete compatible linear-head teacher required')
    model.requires_grad_(False);model.eval()
    rows={}
    for split in ('train','val'):
        if core.sha(c[split]['path'])!=c[split]['sha256'] or ck['config'][split+'_sha256']!=c[split]['sha256']:
            raise ValueError('Teacher/TRAIN/SELECT manifest mismatch')
        rows[split]=core.read_manifest(c[split]['path'])
        if len(rows[split])!=c[split]['rows']:raise ValueError('Membership count mismatch')
    core.split_audit(rows['train'],rows['val'])
    provenance=json.loads(Path(source_provenance).read_text())
    if provenance.get('source_csv_sha256')!=core.sha(source_csv) or provenance.get('train_manifest_sha256')!=c['train']['sha256'] or not provenance.get('source_reference'):
        raise ValueError('Canonical TRAIN source provenance required')
    with Path(source_csv).open(newline='') as f:sr=list(csv.DictReader(f))
    mapping={str(Path(r['image']).resolve()):r['source'] for r in sr}
    real_paths={r['image'] for r in rows['train'] if r['label']==0}
    if len(mapping)!=len(sr) or set(mapping)!=real_paths or not set(mapping.values())<=set(c['source_codes']):
        raise ValueError('Source mapping must cover every TRAIN real exactly; no source guessing or missing rows')
    counts={s:sum(v==s for v in mapping.values()) for s in c['source_codes']}
    if counts!=c['expected_train_real_source_counts']:raise ValueError('Source counts differ from audited canonical TRAIN mapping')
    sources=np.array([c['source_codes'][mapping[r['image']]] if r['label']==0 else -1 for r in rows['train']],dtype=np.int64)
    if any(int((sources==code).sum())<c['auxiliary_per_source'] for code in c['source_codes'].values()):
        raise ValueError('Missing source support')
    fit_inventory={r['image']:r['image_sha256'] for r in ck['fit_membership']}
    if set(fit_inventory)!={r['image'] for s in rows for r in rows[s]}:raise ValueError('Teacher fit inventory differs')
    out=core.fresh_dir(out);file_names=[];seen={};teacher_scores={};max_difference=0.
    model=model.to(device).eval()
    for split in ('train','val'):
        for row in rows[split]:
            # Native load_pixels verifies these checkpoint-bound bytes on every
            # actual image read, avoiding an extra full image scan on the GPU.
            h=fit_inventory[row['image']];row['image_sha256']=h
            if h in seen and seen[h]!=split:raise ValueError('Cross-split content overlap')
            seen[h]=split
        path=out/f'{split}_features.npy'
        features=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(rows[split]),2048))
        scores=np.empty(len(rows[split]),dtype=np.float32)
        for x,ix in DataLoader(Images(rows[split]),batch_size=c['batch_size'],num_workers=c['workers']):
            x=x.to(device);f=model.encoder.features(x).mean((2,3));z=model.classifier(f).squeeze(1)
            # The complete native forward must match feature+head execution.
            if int(ix[0])==0:
                original=model(x)['logit'];torch.testing.assert_close(z,original,rtol=1e-5,atol=1e-6)
                max_difference=max(max_difference,float((z-original).abs().max()))
            features[ix.numpy()]=f.cpu().numpy();scores[ix.numpy()]=z.cpu().numpy()
        features.flush();del features
        np.save(out/f'{split}_teacher_logits.npy',scores);teacher_scores[split]=scores
        core.write_json(out/f'{split}_rows_PRIVATE.json',rows[split])
        file_names += [path.name,f'{split}_teacher_logits.npy',f'{split}_rows_PRIVATE.json']
    threshold=ck['calibration']['threshold_logit']
    replay=core.metric_report(rows['val'],teacher_scores['val'],threshold,c['min_support'])
    if replay!=v['validation']:raise ValueError('Native teacher validation replay differs; preserve cache failure for inspection')
    reference=objective.fit_train_reference(torch.from_numpy(teacher_scores['train'][sources>=0]),c['quantiles'],c['temperature'])
    core.write_json(out/'reference.json',{k:v.tolist() if torch.is_tensor(v) else v for k,v in reference.items()})
    np.save(out/'real_sources.npy',sources)
    torch.save({k:v.detach().cpu() for k,v in model.classifier.state_dict().items()},out/'teacher_head.pt')
    file_names+=['reference.json','real_sources.npy','teacher_head.pt']
    metadata={'format':'SOURCE_TAIL_FEATURE_CACHE_v1','fixture':False,'source_hashes':hashes(),
              'teacher_checkpoint_sha256':core.sha(checkpoint),'teacher_run':str(teacher_run.resolve()),
              'source_csv_sha256':core.sha(source_csv),'source_provenance_sha256':core.sha(source_provenance),
              'train_real_source_counts':counts,
              'manifest_sha256':{s:c[s]['sha256'] for s in rows},'rows':{s:len(rows[s]) for s in rows},
              'files':{n:core.sha(out/n) for n in file_names},'teacher_validation':replay,
              'forward_max_abs_difference':max_difference,'test_read':False,'cal_read':False}
    core.write_json(out/'CACHE_VERIFIED.json',metadata)

def checked_cache(path,fixture=False):
    path=Path(path);m=json.loads((path/'CACHE_VERIFIED.json').read_text())
    if m['format']!='SOURCE_TAIL_FEATURE_CACHE_v1' or bool(m['fixture'])!=fixture or m['source_hashes']!=hashes():
        raise ValueError('Unverified, stale or wrong-fixture cache')
    for name,digest in m['files'].items():
        if Path(name).name!=name or core.sha(path/name)!=digest:raise ValueError('Cache content changed')
    return m

def predict_head(head,features,device,batch=32):
    out=[];head.eval()
    with torch.inference_mode():
        for start in range(0,len(features),batch):
            x=torch.from_numpy(np.array(features[start:start+batch],copy=True)).to(device)
            out.extend(head(x).squeeze(1).cpu().tolist())
    return np.asarray(out)

def train(c,cache,out,arm,device,resume=False,fixture=False,stop_steps=None):
    cache=Path(cache);meta=checked_cache(cache,fixture);out=Path(out)
    if arm not in ('control','treatment'):raise ValueError('Unknown arm')
    if not fixture and (meta['teacher_checkpoint_sha256']!=c['teacher_checkpoint_sha256'] or meta['manifest_sha256']!={s:c[s]['sha256'] for s in ('train','val')}):
        raise ValueError('Wrong teacher/dataset cache')
    arrays={s:np.load(cache/f'{s}_features.npy',mmap_mode='r') for s in ('train','val')}
    rows={s:json.loads((cache/f'{s}_rows_PRIVATE.json').read_text()) for s in arrays}
    src=np.load(cache/'real_sources.npy');teacher=np.load(cache/'train_teacher_logits.npy')
    n=len(rows['train']);y=torch.tensor([r['label'] for r in rows['train']],dtype=torch.float32)
    if len(src)!=n or len(teacher)!=n or any(len(arrays[s])!=len(rows[s]) for s in arrays):raise ValueError('Cache alignment invalid')
    if not np.array_equal(src>=0,y.numpy()==0):raise ValueError('Only original reals may have source labels')
    if not fixture and arrays['train'].shape[1]!=2048:raise ValueError('Wrong encoder feature shape')
    head=nn.Linear(arrays['train'].shape[1],1).to(device)
    head.load_state_dict(torch.load(cache/'teacher_head.pt',map_location=device,weights_only=True),strict=True)
    reference=json.loads((cache/'reference.json').read_text())
    loss_fn=objective.SourceTailObjective(reference,tuple(c['source_codes'].values()),
        c['lambda_source_control'] if arm=='control' else c['lambda_source_treatment'],c['lambda_fake'],c['auxiliary_per_source']).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=c['head_lr'],weight_decay=c['weight_decay'])
    weights=core.original_weights(rows['train'],'ids8');pools={s:np.flatnonzero(src==s) for s in c['source_codes'].values()}
    if any(len(p)<c['auxiliary_per_source'] for p in pools.values()):raise ValueError('Insufficient auxiliary support')
    cache_digest=core.sha(cache/'CACHE_VERIFIED.json');epoch=cursor=steps=0;history=[];best=None
    sums={k:0. for k in ('total','classification','source_tail','fake_guard')};aux_counts={}
    if resume:
        if (out/'DONE').exists():raise ValueError('Completed run preserved')
        ck=torch.load(out/'last.pt',map_location=device,weights_only=True)
        if ck['config']!=c or ck['cache_digest']!=cache_digest or ck['arm']!=arm or ck['source_hashes']!=hashes():raise ValueError('Resume binding mismatch')
        head.load_state_dict(ck['head']);opt.load_state_dict(ck['optimizer'])
        epoch,cursor,steps,history,best=ck['epoch'],ck['cursor'],ck['steps'],ck['history'],ck['best']
        sums,aux_counts=ck['loss_sums'],ck['auxiliary_source_joint_counts']
        if best is not None and core.sha(out/'best.pt')!=ck['best_sha256']:raise ValueError('Selected state changed')
    else:core.fresh_dir(out)
    core.write_json(out/'config.json',c);stop=[False]
    def on_stop(*_):stop[0]=True
    handlers={s:signal.signal(s,on_stop) for s in (signal.SIGTERM,signal.SIGINT)}
    def snapshot():
        return {'format':'SOURCE_TAIL_HEAD_v1','arm':arm,'config':c,'source_hashes':hashes(),'cache_digest':cache_digest,
                'teacher_checkpoint_sha256':meta['teacher_checkpoint_sha256'],'head':head.state_dict(),'optimizer':opt.state_dict(),
                'epoch':epoch,'cursor':cursor,'steps':steps,'history':history,'best':best,'fixture':fixture,
                'loss_sums':sums,'auxiliary_source_joint_counts':aux_counts,
                'best_sha256':core.sha(out/'best.pt') if (out/'best.pt').exists() else None}
    try:
        while epoch<c['epochs']:
            order=np.random.default_rng(c['seed']+epoch).permutation(n)
            batches=[order[j:j+c['batch_size']] for j in range(0,n,c['batch_size'])]
            head.train()
            for ix in batches[cursor:]:
                rng=np.random.default_rng(c['seed']+1000003*steps)
                aux=np.concatenate([rng.choice(p,size=c['auxiliary_per_source'],replace=False) for p in pools.values()])
                x=torch.from_numpy(np.array(arrays['train'][ix],copy=True)).to(device)
                ax=torch.from_numpy(np.array(arrays['train'][aux],copy=True)).to(device)
                loss=loss_fn(head(x).squeeze(1),y[ix].to(device),torch.from_numpy(teacher[ix]).to(device),
                             head(ax).squeeze(1),torch.from_numpy(src[aux]).to(device),weights[ix].to(device))
                if not torch.isfinite(loss['total']):raise RuntimeError('Nonfinite loss')
                opt.zero_grad(set_to_none=True);loss['total'].backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):raise RuntimeError('Invalid head gradients')
                torch.nn.utils.clip_grad_norm_(head.parameters(),c['clip_norm']);opt.step();cursor+=1;steps+=1
                for k in sums:sums[k]+=float(loss[k].detach())
                for j in aux:
                    key=str(src[j])+':'+rows['train'][int(j)]['gender_race']
                    aux_counts[key]=aux_counts.get(key,0)+1
                if stop[0] or (stop_steps and steps>=stop_steps):
                    core.atomic_save(out/'last.pt',snapshot());return {'status':'INCOMPLETE_RESUMABLE','steps':steps}
                if steps%500==0:core.atomic_save(out/'last.pt',snapshot())
            z=predict_head(head,arrays['val'],device);yv=np.array([r['label'] for r in rows['val']])
            cal=core.calibrate(z[yv==0],c['target_fpr']);report=core.metric_report(rows['val'],z,cal['threshold_logit'],c['min_support'])
            o=report['overall'];fair=report['axes']['gender_race']['F_FPR']
            feasible=o['auc']>=c['min_auc'] and o['fnr']<=c['max_fnr'] and fair is not None
            key=[-fair,o['auc'],-(epoch+1)] if feasible else None
            record={'epoch':epoch+1,'steps':steps,'originals_once_each':True,'original_count':n,'validation':report,'calibration':cal,'feasible':feasible,
                    'losses_step_mean':{k:v/len(batches) for k,v in sums.items()},'auxiliary_source_joint_counts':dict(aux_counts)}
            history.append(record)
            if feasible and (best is None or key>best['key']):
                best={'epoch':epoch+1,'key':key,'calibration':cal,'validation':report}
                core.atomic_save(out/'best.pt',snapshot())
            epoch+=1;cursor=0;sums={k:0. for k in sums};aux_counts={}
            core.atomic_save(out/'last.pt',snapshot());core.write_json(out/'history.json',history)
            print(json.dumps({'arm':arm,'epoch':epoch,'auc':o['auc'],'fnr':o['fnr'],'joint_F_FPR':fair,'feasible':feasible}),flush=True)
        if best is None:
            result={'status':'COMPLETED_NO_FEASIBLE_CHECKPOINT','arm':arm,'fixture':fixture,'epochs':epoch}
        else:
            ck=torch.load(out/'best.pt',map_location=device,weights_only=True);head.load_state_dict(ck['head'])
            z=predict_head(head,arrays['val'],device);report=core.metric_report(rows['val'],z,best['calibration']['threshold_logit'],c['min_support'])
            if report!=best['validation']:raise RuntimeError('Selected head replay mismatch')
            result={'status':'COMPLETED_VERIFIED','arm':arm,'fixture':fixture,'epochs':epoch,'selected_epoch':best['epoch'],
                    'validation':report,'checkpoint_sha256':core.sha(out/'best.pt'),'cache_digest':cache_digest,'source_hashes':hashes(),'test_read':False}
            core.write_json(out/'validation_scores_PRIVATE.json',{'rows':rows['val'],'logits':z.tolist(),'threshold':best['calibration']['threshold_logit'],
                'min_support':c['min_support'],'checkpoint_sha256':result['checkpoint_sha256'],'manifest_sha256':meta['manifest_sha256']['val'],'split':'validation'})
        core.write_json(out/'RUN_VERIFIED.json',result);(out/'DONE').write_text(result['status']+'\n');return result
    finally:
        for s,h in handlers.items():signal.signal(s,h)

def export(run,teacher_run,cache,out,device):
    run=Path(run);v=json.loads((run/'RUN_VERIFIED.json').read_text())
    if v['status']!='COMPLETED_VERIFIED' or v.get('fixture') or core.sha(run/'best.pt')!=v['checkpoint_sha256']:raise ValueError('No verified deployable candidate')
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    teacher=Path(teacher_run)/'best.pt'
    if core.sha(teacher)!=ck['teacher_checkpoint_sha256'] or ck['source_hashes']!=hashes():raise ValueError('Teacher/source mismatch')
    cache=Path(cache);checked_cache(cache)
    if core.sha(cache/'CACHE_VERIFIED.json')!=ck['cache_digest']:raise ValueError('Wrong export cache')
    model,teacher_ck=core.load_trained(teacher,device);model.classifier.load_state_dict(ck['head'],strict=True)
    model.requires_grad_(False);model.eval()
    rows=json.loads((cache/'val_rows_PRIVATE.json').read_text())
    scores=core.predict(model,rows,teacher_ck['config'],device)
    report=core.metric_report(rows,scores,ck['best']['calibration']['threshold_logit'],ck['config']['min_support'])
    if report!=v['validation']:raise RuntimeError('Full native validation replay differs from cached-head report; inspect numerical/preprocessing mismatch before deployment')
    if Path(out).exists():raise ValueError('Preserve existing export')
    torch.save({'format':'SOURCE_TAIL_STUDENT_v1','model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'calibration':ck['best']['calibration'],'teacher_checkpoint_sha256':ck['teacher_checkpoint_sha256'],
                'source_hashes':hashes(),'input_processing':'native teacher RGB/direct224/(x-.5)/.5','source_labels_needed_at_inference':False},out)
    core.write_json(str(out)+'.VERIFIED.json',{'status':'PASS','full_native_validation_replay':True,'checkpoint_sha256':core.sha(out),'source_labels_needed_at_inference':False})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','train','export'])
    p.add_argument('--out',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--teacher-run');p.add_argument('--source-csv');p.add_argument('--source-provenance');p.add_argument('--cache')
    p.add_argument('--arm',choices=['control','treatment']);p.add_argument('--resume',action='store_true');p.add_argument('--run')
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;core.seed_all(17);c=config();d=torch.device(a.device)
    if d.type=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):raise ValueError('Exactly one bound GPU required')
    if a.stage=='prepare':prepare(c,a.teacher_run,a.source_csv,a.source_provenance,a.out,d)
    elif a.stage=='train':train(c,a.cache,a.out,a.arm,d,a.resume)
    else:export(a.run,a.teacher_run,a.cache,a.out,d)
