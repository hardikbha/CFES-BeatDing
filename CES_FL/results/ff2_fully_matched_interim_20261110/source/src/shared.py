"""Fully shared FF2 downstream recipe. Controlled objective adaptations only.

All methods share architecture, initial parameters, trainability, optimiser,
batch/order, unweighted base BCE, paired/auxiliary forward exposure, selection,
thresholding and evaluator. Only the registered objective and Ding's declared
TRAIN-derived feature mask differ. Three downstream seeds share one P50 encoder.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SOURCE = Path(__file__).resolve().parents[1]
ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/ff2_fully_matched_20261110_r1')
VERSION = 'FF2_FULLY_MATCHED_v1'
METHODS = ('bce', 'dag', 'daw', 'ding_sfd_gda', 'lh', 'lh_ramp', 'lh_low')
SEEDS = (17, 7, 27)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

core = load('matched_ff2_core', Path(__file__).with_name('core.py'))
ported = load('matched_ff2_losses', Path(__file__).with_name('ported_losses.py'))
ORIGINAL_DETECTOR = core.Detector
ORIGINAL_INITIALISE = core.load_initial
ORIGINAL_STEP = core.train_step

def hashes():
    return {p.name: core.sha(p) for p in [Path(__file__), Path(core.__file__), Path(ported.__file__), SOURCE/'data_cache.py']}

def check(c):
    if c['suite_format'] != VERSION or c['suite_method'] not in METHODS or c['seed'] not in SEEDS:
        raise ValueError('Unknown matched method/seed')
    if c['suite_hashes'] != hashes():raise ValueError('Suite scientific source changed')
    fixed = dict(dataset_id='FairForge-2_public', epochs=10, batch_size=32, workers=4, size=224,
                 encoder_policy='exit', encoder_lr=1e-5, head_lr=.001, weight_decay=.0001,
                 head_type='linear', amp=True, target_fpr=.1, min_train_support=20,
                 min_eval_support=20, group_per_class=4, aux_every=4,
                 train_calibration_size=2048, temperature=1., margin=.5,
                 lambda_fair=.1, lambda_sensitivity=.1, lambda_alignment=.1, sham_weight=.5,
                 error_control='hierarchical_real', alignment_warmup_epochs=0,
                 common_classification='unweighted_BCE', ding_lambda=.005, sinkhorn_blur=.0005,
                 dag_alpha=.5, daw_inner_alpha=.9, daw_outer_alpha=.5)
    for key, value in fixed.items():
        if c.get(key) != value:raise ValueError('Shared setting differs: ' + key)
    if c['axes'] != ['gender_race']:raise ValueError('Shared auxiliary axis is fixed to the eight joint groups')
    for key,hkey in [('train','train_sha256'),('val','val_sha256'),('init_checkpoint','init_sha256')]:
        if core.sha(c[key]) != c[hkey]:raise ValueError('Input changed: ' + key)
    if c['source_sha256'] != core.sha(core.__file__):raise ValueError('Historical functional core changed')
    return c

class SharedDetector(ORIGINAL_DETECTOR):
    def __init__(self, hidden=128, gate_hidden=64, policy='exit', head_type='linear'):
        super().__init__(hidden, gate_hidden, policy, head_type)
        if head_type != 'linear':raise ValueError('One shared linear architecture required')
        self.register_buffer('feature_mask', torch.ones(2048))
    def forward(self, x):
        f = self.encoder.features(x) * self.feature_mask[None,:,None,None]
        z = self.classifier(f.mean((2,3))).squeeze(1)
        spatial = (f * self.classifier.weight[0][None,:,None,None]).sum(1)
        return {'logit': z, 'local_map': spatial, 'global_contribution': torch.zeros_like(z),
                'bias': self.classifier.bias[0]}

def initialise(c):
    model = ORIGINAL_INITIALISE(c)
    if c['suite_method'] == 'ding_sfd_gda':
        artifact=Path(c['sfd_file'])
        binding=json.loads(artifact.with_name(artifact.name+'.sha256.json').read_text())
        if binding['sha256']!=core.sha(artifact):raise ValueError('Frozen SFD artifact changed')
        report = json.loads(Path(c['sfd_file']).read_text())
        if report['status'] != 'PASS' or report['train_sha256'] != c['train_sha256'] or report['init_sha256'] != c['init_sha256']:
            raise ValueError('SFD population/initializer differs')
        if report['suite_hashes']!=hashes() or report['train_images']!=218266 or report['group_names']!=c['group_names']:
            raise ValueError('SFD recipe/coverage differs')
        model.feature_mask[report['channels']] = 0
    return model

def response_factor(c, epoch):
    method = c['suite_method']
    if method == 'lh':return 1.
    if method not in ('lh_ramp','lh_low'):return 0.
    if epoch < 2:return 0.
    return 1/3 if method == 'lh_low' else min(1., (epoch-1)/3)

def symmetric_logits(z):return torch.stack((-z.float()/2, z.float()/2), 1)

def method_objective(c, z, y, groups, fair, sens, sham, align, gda, has_aux, epoch):
    cls = F.binary_cross_entropy_with_logits(z.float(), y.float())
    method = c['suite_method'];zero = z.float().sum()*0
    if method == 'dag':main = ported.dag_fdd_loss(z,y,c['dag_alpha'])[0]
    elif method == 'daw':main = ported.daw_fdd_loss(z,y,groups,c['daw_inner_alpha'],c['daw_outer_alpha'])[0]
    else:main = cls
    total = main
    if method == 'ding_sfd_gda':total = total+c['aux_every']*c['ding_lambda']*gda
    if method in ('lh','lh_ramp','lh_low'):
        total = total+c['aux_every']*c['lambda_fair']*fair
        total = total+response_factor(c,epoch)*(c['lambda_sensitivity']*sens+
                    c['lambda_sensitivity']*c['sham_weight']*sham+c['lambda_alignment']*align)
    return {'total':total,'classification':cls,'classification_objective':main,
            'error_fairness':fair,'sensitivity':sens,'sham':sham,'alignment':align,'gda':gda}

_SINKHORN = None
def sinkhorn():
    global _SINKHORN
    if _SINKHORN is None:
        from geomloss import SamplesLoss
        _SINKHORN=SamplesLoss(loss='sinkhorn',p=2,blur=.0005)
    return _SINKHORN

def train_step(model, b, aux, weights, c, arm, threshold, device, epoch=0):
    x,y=b['x'].to(device),b['y'].to(device).float()
    with core.autocast(device,c):
        z=model(x)['logit'];zero=z.float().sum()*0
        real=y==0;paired=int(real.sum());sens=zero;sham=zero;fair=zero;align=zero;gda=zero
        # Identical forward exposure, in identical order, for every method.
        if real.any():
            zs=model(b['s'][real.cpu()].to(device))['logit'];zt=model(b['t'][real.cpu()].to(device))['logit']
            sens=core.sensitivity(core.response(zs,zt),c['margin'])
            sham=(torch.asinh(zs.float())-torch.asinh(z[real].float())).abs().mean()
        if aux is not None:
            za=model(aux['x'].to(device))['logit'];ay=aux['y'].to(device);ag=aux['group'].to(device)
            fair=core.hierarchical_real_error(za,ay,ag,aux['priors'],aux['hierarchy'],threshold,c['temperature'])
            zs=model(aux['s'].to(device))['logit'];zt=model(aux['t'].to(device))['logit']
            rr=core.response(zs,zt)
            sens=sens+c['aux_every']*core.sensitivity(rr,c['margin'])
            sham=sham+c['aux_every']*(torch.asinh(zs.float())-torch.asinh(za[ay==0].float())).abs().mean()
            align=c['aux_every']*core.response_alignment(rr,ag[ay==0])
            gi=core.seed_for(c['seed'],epoch,tuple(aux['indices']),'gda-reference')%len(aux['groups'])
            # Sinkhorn distance arithmetic must stay FP32 even when the common
            # encoder/classifier forward uses BF16 autocast.
            with torch.autocast(device_type=device.type,enabled=False):
                gda=sinkhorn()(symmetric_logits(z),symmetric_logits(za[ag==gi]).detach())
        main_groups=b['group'].to(device)
        losses=method_objective(c,z,y,main_groups,fair,sens,sham,align,gda,aux is not None,epoch)
    return losses,paired

class CommonOriginals(core.Originals):
    def __getitem__(self,i):
        out=super().__getitem__(i);out['group']=self.c['group_names'].index(self.rows[i]['gender_race']);return out

def make_loader(rows,c,epoch=0,pairs=False,shuffle=False):
    ds=CommonOriginals(rows,c,epoch,pairs)
    return DataLoader(ds,batch_size=c['batch_size'],shuffle=shuffle,num_workers=c['workers'],
                      pin_memory=False,drop_last=False,generator=torch.Generator().manual_seed(c['seed']+epoch))

_INPUT_CACHE = None
def checked_inputs(c,preflight):
    global _INPUT_CACHE
    receipt=json.loads((Path(preflight)/'PREFLIGHT.json').read_text())
    audit_path=ROOT/'shared_input_audit/PREFLIGHT.json'
    audit=json.loads(audit_path.read_text())
    if receipt['status']!='PASS' or receipt['config_digest']!=core.digest(c) or receipt['source_sha256']!=core.sha(core.__file__):
        raise ValueError('Wrong data/configuration binding')
    if audit['cache_builder_sha256']!=core.sha(SOURCE/'data_cache.py') or not audit['fresh_all_image_hashes'] or not audit['fresh_full_decode']:
        raise ValueError('Full current input audit missing')
    if _INPUT_CACHE is None:
        cache=Path(audit['row_cache_path'])
        if core.sha(cache)!=audit['row_cache_sha256']:raise ValueError('Row cache changed')
        _INPUT_CACHE=json.loads(cache.read_text())
    train,val=_INPUT_CACHE['train'],_INPUT_CACHE['val']
    if len(train)!=218266 or len(val)!=31107:raise ValueError('Wrong dataset population')
    return train,val,receipt

def configure():
    core.Detector=SharedDetector
    core.load_initial=initialise
    core.train_step=train_step
    core.original_weights=lambda rows,axis:torch.ones(len(rows),dtype=torch.float32)
    core.make_loader=make_loader
    core.checked_inputs=checked_inputs

def initial_fingerprint(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def train(c,preflight,out,qualification,device,resume=False,fixture=False,stop_after=None):
    if not fixture:check(c)
    configure()
    root=Path(out)
    # The core lifecycle saves the full suite configuration and therefore binds
    # method/source/recipe for exact resume, while its immutable functions remain intact.
    if not resume:
        model=initialise(c)
        initial={'status':'PASS','method':c['suite_method'],'seed':c['seed'],
                 'parameter_fingerprint':initial_fingerprint(model),'parameters':model.counts(),
                 'masked_channels':torch.nonzero(model.feature_mask==0).flatten().tolist(),
                 'suite_hashes':hashes(),'test_read':False}
        del model
    result=core.train(c,preflight,root,'D',device,qualification=qualification,resume=resume,
                      fixture=fixture,stop_after=stop_after,budget_hours=c['max_gpu_hours'])
    if not resume:core.write_json(root/'INITIAL_STATE.json',initial)
    if result['status']=='COMPLETED_VERIFIED':
        result.update(suite_format=VERSION,method=c['suite_method'],suite_hashes=hashes(),
                      common_axis='gender_race',classification='unweighted BCE; CVaR variants declared',
                      shared_pretraining='one frozen seed17 P50 encoder; downstream-seed replication')
        core.write_json(root/'RUN_VERIFIED.json',result)
        ck=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
        core.write_json(root/'FROZEN_CHECKPOINT.json',{'status':'FROZEN','checkpoint_sha256':core.sha(root/'best.pt'),
                        'suite_config_digest':core.digest(c),'suite_hashes':hashes(),
                        'selected_epoch':result['selected_epoch'],'threshold_logit':ck['calibration']['threshold_logit'],
                        'test_read':False})
    return result

def evaluate(c,root,out,device):
    check(c);configure();root=Path(root);out=core.fresh_dir(out)
    verified=json.loads((root/'RUN_VERIFIED.json').read_text())
    frozen=json.loads((root/'FROZEN_CHECKPOINT.json').read_text())
    if verified['status']!='COMPLETED_VERIFIED' or frozen['checkpoint_sha256']!=core.sha(root/'best.pt'):
        raise ValueError('A complete frozen checkpoint is required before TEST')
    if core.sha(c['test'])!=c['test_sha256']:raise ValueError('Frozen TEST manifest changed')
    ck=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
    if ck['config']!=c:raise ValueError('Selected model configuration differs')
    model=SharedDetector(c['hidden'],c['gate_hidden'],c['encoder_policy'],c['head_type']).to(device)
    model.load_state_dict(ck['model'],strict=True)
    rows=core.read_manifest(c['test'])
    if len(rows)!=87564:raise ValueError('Wrong frozen public TEST population')
    scores=core.predict(model,rows,c,device)
    threshold=ck['calibration']['threshold_logit']
    report=core.metric_report(rows,scores,threshold,c['min_eval_support'])
    core.write_json(out/'metrics.json',{'method':c['suite_method'],'seed':c['seed'],
        'calibrated':report,'fixed_probability_0_5':core.metric_report(rows,scores,0.,c['min_eval_support']),
        'checkpoint_sha256':frozen['checkpoint_sha256'],'manifest_sha256':c['test_sha256'],
        'threshold_source':'frozen selected SELECT-real10%FPR threshold; no TEST recalibration',
        'suite_hashes':hashes(),'test_used_for_selection':False})
    core.write_json(out/'scores_PRIVATE.json',{'rows':rows,'logits':scores.tolist(),'threshold':threshold,
        'checkpoint_sha256':frozen['checkpoint_sha256'],'manifest_sha256':c['test_sha256']})
    return report

def sfd(c,preflight,out,device):
    if Path(out).exists():raise ValueError('Preserve existing SFD artifact')
    configure();rows,_,_=core.checked_inputs(c,preflight)
    model=ORIGINAL_INITIALISE(c).to(device).eval();totals=torch.zeros(2048,device=device);batches=0;seen=Counter()
    order=np.random.default_rng(17).permutation(len(rows)).tolist()
    loader=DataLoader(CommonOriginals(rows,c,pairs=False),batch_sampler=[order[i:i+32] for i in range(0,len(rows),32)],
                      num_workers=c['workers'])
    with torch.inference_mode():
        for b in loader:
            feat=model.encoder.features(b['x'].to(device)).float().flatten(2)
            values=ported.snnl_per_channel(feat,b['group'].to(device))
            if not torch.isfinite(values).all():raise RuntimeError('Nonfinite SFD statistics')
            totals+=values;batches+=1;seen.update(b['index'].tolist())
            if batches%200==0:print(json.dumps({'stage':'TRAIN-only SFD','batches':batches}),flush=True)
    if len(seen)!=len(rows) or any(v!=1 for v in seen.values()):raise RuntimeError('SFD must cover each TRAIN original once')
    average=(totals/batches).cpu().numpy();channels,threshold,n=ported.select_sfd_channels(average,prc=2.)
    if not channels:raise RuntimeError('No SFD channel selected')
    core.write_json(out,{'status':'PASS','channels':channels,'threshold':threshold,'requested_n':n,
        'feature_channels':2048,'train_sha256':c['train_sha256'],'init_sha256':c['init_sha256'],
        'batches':batches,'train_images':len(rows),'group_names':c['group_names'],
        'suite_hashes':hashes(),'test_read':False,'shared_representation_adaptation':True})
    artifact=Path(out)
    core.write_json(artifact.with_name(artifact.name+'.sha256.json'),{'status':'FROZEN','sha256':core.sha(artifact),'test_read':False})

def dry(c,preflight,qualification,out,device):
    check(c);configure();core.check_qualification(c,qualification)
    rows,_,_=core.checked_inputs(c,preflight);active,_=core.group_inventory(rows,c)
    model=initialise(c).to(device);opt=core.build_optimizer(model,c)
    ix=[next(i for i,r in enumerate(rows) if r['label']==0)]+list(range(31))
    ds=CommonOriginals(rows,c,epoch=0,pairs=True);items=[ds[i] for i in ix]
    b={k:torch.stack([x[k] for x in items]) if torch.is_tensor(items[0][k]) else torch.tensor([x[k] for x in items]) for k in items[0]}
    aux=core.auxiliary(rows,active,c,0,0)
    reals=sorted([r for r in rows if r['label']==0],key=lambda r:core.seed_for(c['seed'],'train-calibration',r['image']))[:2048]
    threshold=core.calibrate(core.predict(model,reals,c,device),.1)['threshold_logit'];model.train()
    bn={n:x.clone() for n,x in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
    losses,count=train_step(model,b,aux,torch.ones(len(rows)),c,'D',threshold,device)
    core.finite_step(losses,model,opt)
    if any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in bn.items()):raise RuntimeError('BN changed')
    exposure=hashlib.sha256()
    for prefix,values in [('main',b),('aux',aux)]:
        for key in ['x','s','t','y','group','index']:
            value=values.get(key)
            if torch.is_tensor(value):
                exposure.update((prefix+key+str(tuple(value.shape))).encode());exposure.update(value.detach().cpu().contiguous().numpy().tobytes())
    exposure.update(core.digest(aux['indices']).encode())
    core.write_json(out,{'status':'PASS','method':c['suite_method'],'seed':c['seed'],'device':str(device),
        'suite_hashes':hashes(),'parameters':model.counts(),'initial_parameter_fingerprint':initial_fingerprint(initialise(c)),
        'losses':{k:float(v.detach()) for k,v in losses.items()},'real_pairs':count,
        'optimizer_steps_in_discarded_model':1,'test_read':False,'bn_frozen':True,
        'data_forward_fingerprint':exposure.hexdigest()})

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['dry','train','evaluate','sfd'])
    p.add_argument('--config',required=True);p.add_argument('--preflight');p.add_argument('--qualification')
    p.add_argument('--out',required=True);p.add_argument('--training');p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--resume',action='store_true');a=p.parse_args();c=json.loads(Path(a.config).read_text())
    check(c);torch.set_num_threads(2);device=torch.device(a.device)
    if a.command=='dry':dry(c,a.preflight,a.qualification,a.out,device)
    elif a.command=='sfd':sfd(c,a.preflight,a.out,device)
    elif a.command=='train':train(c,a.preflight,a.out,a.qualification,device,a.resume)
    else:evaluate(c,a.training,a.out,device)

if __name__=='__main__':main()
