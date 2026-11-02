"""Versioned original-only Ding architecture on FairForge-2 public.

Native sampling/global batches/224px/SGD are preserved. GDA is explicitly a
corrected class-conditioned empirical-OT variant, NOT the detached archived
fairness implementation. See the run proposal for all disclosed differences.
No TEST inference is implemented in the training command.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as grad_all_gather
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / 'DINGREPRO_LIN/code/Ding_mod'
sys.path.insert(0, str(NATIVE))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


Xception = module(NATIVE/'utils/xception.py', 'ding_native_xception').Xception
common = module(Path(__file__).with_name('ces_fl_diagram_model_v021.py'), 'ding_report_common')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def log(event, **fields):
    print(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), event=event, **fields)), flush=True)


def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Detector(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.backbone = Xception(dict(mode='adjust_channel', num_classes=2, inc=3, dropout=False))
        self.head_only = False
        if weights:
            state = torch.load(weights, map_location='cpu', weights_only=True)
            state = {k: (v[..., None, None] if 'pointwise' in k and v.ndim == 2 else v)
                     for k, v in state.items() if 'fc' not in k}
            result = self.backbone.load_state_dict(state, strict=False)
            if result.unexpected_keys:
                raise ValueError(result)
            expected = {'adjust_channel.0.weight', 'adjust_channel.0.bias', 'adjust_channel.1.weight',
                        'adjust_channel.1.bias', 'adjust_channel.1.running_mean', 'adjust_channel.1.running_var',
                        'last_linear.weight', 'last_linear.bias'}
            if set(result.missing_keys) != expected:
                raise ValueError(('Unexpected initialization missing keys', result))

    def forward(self, x):
        return self.backbone.classifier(self.backbone.features(x))

    def policy(self, head_only):
        self.head_only = head_only
        for p in self.parameters():
            p.grad = None
            p.requires_grad_(not head_only)
        if head_only:
            for p in self.backbone.last_linear.parameters():
                p.requires_grad_(True)
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.head_only:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self


def mask_channels(model, channels):
    from torch.nn.utils import prune
    layer = model.backbone.adjust_channel[0]
    mask = torch.ones_like(layer.weight)
    mask[channels] = 0
    prune.custom_from_mask(layer, 'weight', mask)


def snnl(features, groups):
    """Vectorized, signed native SFD cosine loss (self in numerator only)."""
    x = F.normalize(features.flatten(2).permute(1, 0, 2), dim=-1)
    e = torch.exp(-(1 - torch.bmm(x, x.transpose(1, 2))))
    same = (groups[:, None] == groups[None, :]).to(e.dtype)
    off = 1 - torch.eye(len(groups), device=e.device, dtype=e.dtype)
    return -torch.log(((e*same).sum(-1)+1e-9)/(e*off).sum(-1)).mean(-1)


def sfd_features(model, x):
    # Preserve archived SFD helper, rather than silently substitute features().
    m = model.backbone
    return m.adjust_channel[0](nn.Sequential(*list(m.children())[:-2])(x))


def ot_loss(main_logits, main_y, aux_logits, aux_y, epsilon):
    from geomloss import SamplesLoss
    if not torch.is_grad_enabled() or not main_logits.requires_grad or not aux_logits.requires_grad:
        raise ValueError('Corrected GDA requires live logits')
    criterion = SamplesLoss('sinkhorn', p=2, blur=math.sqrt(epsilon), diameter=1.,
                            debias=False, backend='tensorized')
    main = main_logits.softmax(-1)[:, 1]
    aux = aux_logits.softmax(-1)[:, 1]
    terms = []
    for y in (0, 1):
        a, b = aux[aux_y == y, None], main[main_y == y, None]
        if min(len(a), len(b)) < 2:
            raise ValueError('Missing class-conditional OT support')
        terms.append(criterion(a, b))
    return sum(terms)


def gather(x, gradient=False):
    """Unequal per-rank batches, no duplicated padding observations."""
    if not dist.is_initialized():
        return x
    n = torch.tensor([len(x)], device=x.device, dtype=torch.int64)
    ns = [torch.zeros_like(n) for _ in range(dist.get_world_size())]
    dist.all_gather(ns, n)
    sizes = [int(v.item()) for v in ns]
    padding = x.new_zeros((max(sizes)-len(x),)+x.shape[1:])
    padded = torch.cat([x, padding], 0)
    if gradient:
        parts = grad_all_gather(padded)
    else:
        parts = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(parts, padded)
    return torch.cat([v[:n] for v, n in zip(parts, sizes)])


class Images(Dataset):
    def __init__(self, rows, evaluation=False):
        self.rows = rows
        geometry = [transforms.Resize(224), transforms.CenterCrop(224)] if evaluation else [transforms.Resize((224, 224))]
        self.transform = transforms.Compose(geometry + [transforms.ToTensor(), transforms.Normalize((.5,)*3, (.5,)*3)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        # Tuple sampler keys preserve explicit role/y/group alignment.
        index, role = key if isinstance(key, tuple) else (key, 0)
        r = self.rows[index]
        with Image.open(r['image']) as image:
            x = self.transform(image.convert('RGB'))
        return x, int(r['label']), int(r['native_group']), role, index


class NativeBatches(Sampler):
    def __init__(self, rows, pair_batch, seed_value, rank=0, world=1, auxiliary=False):
        self.rows, self.pair_batch, self.seed_value = rows, pair_batch, seed_value
        self.rank, self.world, self.auxiliary = rank, world, auxiliary
        self.real = np.array([i for i, r in enumerate(rows) if r['label'] == 0])
        self.fake = np.array([i for i, r in enumerate(rows) if r['label'] == 1])
        self.groups = {(g,y): np.array([i for i,r in enumerate(rows) if r['label']==y and r['native_group']==g])
                       for g in range(8) for y in (0,1)}
        if any(not len(v) for v in self.groups.values()):
            raise ValueError('Native eight groups require both classes')

    def __len__(self):
        return math.ceil(len(self.fake)/self.pair_batch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed_value)
        fake = rng.permutation(self.fake)
        for start in range(0, len(fake), self.pair_batch):
            ff = fake[start:start+self.pair_batch]
            rr = rng.choice(self.real, len(ff), replace=True)
            sl = slice(self.rank, len(ff), self.world)
            # Local ordering is real/main, fake/main, real/aux, fake/aux.
            keys = [(int(i), 0) for i in np.r_[rr[sl], ff[sl]]]
            if self.auxiliary:
                g = int(rng.integers(8))
                ar = rng.choice(self.groups[g,0], self.pair_batch, replace=True)
                af = rng.choice(self.groups[g,1], self.pair_batch, replace=True)
                keys += [(int(i),1) for i in np.r_[ar[self.rank::self.world], af[self.rank::self.world]]]
            if not keys:
                raise ValueError('Global tail has no observations on a rank')
            yield keys


class SequentialBatches(Sampler):
    def __init__(self, n, size, rank, world, shuffled_seed=None):
        self.n, self.size, self.rank, self.world, self.shuffled_seed = n,size,rank,world,shuffled_seed

    def __len__(self):
        return math.ceil(self.n/self.size)

    def __iter__(self):
        ids = np.arange(self.n) if self.shuffled_seed is None else np.random.default_rng(self.shuffled_seed).permutation(self.n)
        for start in range(0,self.n,self.size):
            local = ids[start:start+self.size][self.rank::self.world]
            if not len(local):
                raise ValueError('Empty distributed inference tail; no padded duplicate rows permitted')
            yield local.tolist()


def read_inputs(c, stat=True):
    splits = {}
    groups = {}
    for split in ('train','val','test'):
        path = c['manifests'][split]
        if sha(path) != c['input_hashes'][split]:
            raise ValueError(('Manifest changed', split))
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != c['counts'][split]:
            raise ValueError(('Count changed', split))
        seen = set()
        for r in rows:
            r['label'] = int(r['label'])
            if r['label'] not in (0,1) or r['image'] in seen:
                raise ValueError('Invalid/duplicate row')
            seen.add(r['image'])
            if not r['identity_ids'] or not r['source_ids']:
                raise ValueError('Missing provenance')
        if stat:
            # Bound metadata concurrency on the shared filesystem; no image decoding here.
            with ThreadPoolExecutor(max_workers=16) as executor:
                for r, present in zip(rows, executor.map(lambda r: Path(r['image']).is_file(), rows)):
                    if not present:
                        raise ValueError(('Missing image', r['image']))
            log('INPUT_FILES_CHECKED',split=split,rows=len(rows),test_pixels_read=False)
        if {r['label'] for r in rows} != {0,1}:
            raise ValueError('Single-class split')
        splits[split] = rows
    if sha(c['native_train']) != c['input_hashes']['native_train']:
        raise ValueError('Native training schema changed')
    with open(c['native_train'], newline='') as f:
        native = list(csv.DictReader(f))
    ni = {r['img_path']:r for r in native}
    if len(ni)!=len(native) or {(r['image'],r['label']) for r in splits['train']} != {(r['img_path'],int(r['label'])) for r in native}:
        raise ValueError('Native/canonical TRAIN membership mismatch')
    for r in splits['train']:
        r['native_group'] = int(ni[r['image']]['intersec_label'])
        if not 0 <= r['native_group'] < 8:
            raise ValueError('Invalid native group')
        groups.setdefault(r['native_group'],set()).add(r['gender_race'])
    if len(groups)!=8 or any(len(v)!=1 for v in groups.values()) or len({next(iter(v)) for v in groups.values()})!=8:
        raise ValueError('Native/canonical demographic mapping mismatch')
    for s in ('val','test'):
        for r in splits[s]:
            r['native_group'] = -1  # not used in gradient/auxiliary sampling
    overlap = {}
    for a,b in (('train','val'),('train','test'),('val','test')):
        for field in ('image','identity_ids','source_ids'):
            left = {v for r in splits[a] for v in r[field].split(';') if v}
            right = {v for r in splits[b] for v in r[field].split(';') if v}
            overlap[f'{a}:{b}:{field}'] = len(left&right)
    if any(overlap.values()):
        raise ValueError(('Recorded split overlap', overlap))
    if sha(c['imagenet_weights']) != c['input_hashes']['imagenet_weights']:
        raise ValueError('Initialization weights changed')
    return splits, dict(overlaps=overlap, native_groups={str(k):list(v) for k,v in groups.items()},
                        class_counts={s:dict(Counter(r['label'] for r in rows)) for s,rows in splits.items()},
                        hidden_person_or_donor_independence_verified=False, test_pixels_read=False)


@torch.no_grad()
def validation(model, rows, c, device, rank, world):
    model.eval()
    batches = SequentialBatches(len(rows),c['validation_batch'],rank,world)
    loader = DataLoader(Images(rows,True),batch_sampler=batches,num_workers=c['workers_per_rank'],pin_memory=True)
    scores, indices = [], []
    for x,_,_,_,idx in loader:
        z = model(x.to(device,non_blocking=True))
        scores.extend(gather((z[:,1]-z[:,0]).double()).cpu().tolist())
        indices.extend(gather(idx.to(device)).cpu().tolist())
    order = np.argsort(indices)
    if sorted(indices)!=list(range(len(rows))):
        raise ValueError('Validation missing/duplicate rows')
    z = np.asarray(scores)[order]
    cal = common.calibrate(z[np.array([r['label'] for r in rows])==0],c['target_fpr'])
    result = common.metrics(rows,z,cal['threshold_logit'])
    return z, cal, result


def save_ckpt(path, model, optimizer, epoch, stage, c, cal, result):
    obj = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,stage=stage,
               config=c,calibration=cal,validation=result,channels=getattr(model,'selected_channels',[]),
               source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               source_hashes=source_hashes(), test_used=False)
    temp = path.with_suffix('.tmp')
    torch.save(obj,temp)
    os.replace(temp,path)


def source_hashes():
    paths = [Path(__file__),Path(__file__).with_name('ces_fl_diagram_model_v021.py'),
             NATIVE/'utils/xception.py',NATIVE/'utils/registry.py',NATIVE/'SFD.py',NATIVE/'detectors/xception_detector.py',
             ROOT/'CES_FL/scripts/run_ding_ff2_ssh.py',ROOT/'CES_FL/tests/test_ding_ff2_public.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def train_stage(model, stage, train, val, c, out, device, rank, world, smoke=False):
    stage_dir = out/stage
    if rank==0:
        stage_dir.mkdir(exist_ok=False)
    dist.barrier()
    head = stage=='gda_corrected'
    model.policy(head)
    if not head:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                               lr=c['lr'],momentum=c['momentum'],weight_decay=c['weight_decay'])
    wrapped = DDP(model, device_ids=[device.index],broadcast_buffers=False)
    steps = 0
    best = -math.inf
    if rank==0:
        log('STAGE_START',stage=stage,total_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            global_pair_batch=c['gda_pair_batch'] if head else c['classification_pair_batch'])
    for epoch in range(1, (1 if smoke else c['epochs'])+1):
        wrapped.train()
        sampler = NativeBatches(train,c['gda_pair_batch'] if head else c['classification_pair_batch'],
                                c['seed']+1000*int(head)+epoch,rank,world,head)
        loader = DataLoader(Images(train),batch_sampler=sampler,num_workers=c['workers_per_rank'],
                            pin_memory=True,persistent_workers=c['workers_per_rank']>0)
        start = time.time()
        accum = np.zeros(3)
        for batch,(x,y,g,role,_) in enumerate(loader,1):
            x,y,role = x.to(device,non_blocking=True),y.to(device,non_blocking=True),role.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            z = wrapped(x)
            if not torch.isfinite(z).all():
                raise FloatingPointError('Nonfinite logits')
            main = role==0
            n = torch.tensor(int(main.sum()),device=device)
            dist.all_reduce(n)
            ce = F.cross_entropy(z[main],y[main],reduction='sum')*world/n
            fair = z.new_zeros(())
            if head:
                # Autograd all-gather supports unequal main tail and fixed auxiliary batches.
                zz, yy, rr = gather(z,True),gather(y),gather(role)
                fair = ot_loss(zz[rr==0],yy[rr==0],zz[rr==1],yy[rr==1],c['epsilon'])
            loss = ce+c['lambda_fair']*fair
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite objective')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
                raise FloatingPointError('Missing/nonfinite parameter gradients')
            optimizer.step()
            steps += 1
            values = torch.stack([ce.detach(),fair.detach(),loss.detach()])
            dist.all_reduce(values)
            accum += (values/world).cpu().numpy()
            if rank==0 and (batch==1 or batch%c['log_every']==0):
                log('TRAIN_STEP',stage=stage,epoch=epoch,batch=batch,batches=len(sampler),global_step=steps,
                    ce=float(values[0]/world),fair=float(values[1]/world),loss=float(values[2]/world),
                    elapsed_seconds=time.time()-start,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            if smoke:
                break
        if smoke:
            break
        # SyncBN synchronized training buffers; evaluation itself has no collective forward.
        z,cal,result = validation(model,val,c,device,rank,world)
        auc = result['overall']['auc']
        if rank==0:
            rec = dict(epoch=epoch,stage=stage,steps=steps,mean_ce=float(accum[0]/len(sampler)),
                       mean_fair=float(accum[1]/len(sampler)),seconds=time.time()-start,
                       calibration=cal,validation=result,test_used=False)
            with (stage_dir/'history.jsonl').open('a') as f:
                f.write(json.dumps(rec,allow_nan=False)+'\n')
            save_ckpt(stage_dir/'last.pt',model,optimizer,epoch,stage,c,cal,result)
            if auc>best:
                best=auc
                save_ckpt(stage_dir/'best.pt',model,optimizer,epoch,stage,c,cal,result)
                np.savez_compressed(stage_dir/'validation_best.npz',logits=z)
            # Native-compatible per-epoch state dictionaries, never removed/overwritten.
            torch.save(model.state_dict(),stage_dir/f'{epoch-1}.pth')
            log('EPOCH_COMPLETE',stage=stage,epoch=epoch,validation_auc=auc,
                achieved_validation_fpr=cal['achieved_fpr'],gender_race=result['axes']['gender_race'])
        best=max(best,auc)
        dist.barrier()
    if rank==0:
        write(stage_dir/'DONE.json',dict(completed_epochs=1 if smoke else c['epochs'],steps=steps,smoke=smoke))
    dist.barrier()
    del wrapped
    return model


def run_sfd(model, rows, c, out, device, rank, world):
    model.eval()
    sampler = SequentialBatches(len(rows),c['sfd_batch'],rank,world,c['seed']+2000)
    loader = DataLoader(Images(rows),batch_sampler=sampler,num_workers=c['workers_per_rank'],pin_memory=True)
    totals = torch.zeros(512,device=device,dtype=torch.float64)
    for batch,(x,_,g,_,_) in enumerate(loader,1):
        with torch.no_grad():
            features = gather(sfd_features(model,x.to(device)))
            gg = gather(g.to(device))
            values = snnl(features,gg)
        if not torch.isfinite(values).all():
            raise FloatingPointError('Invalid SFD scores')
        totals += values.double()
        if rank==0 and (batch==1 or batch%100==0):
            log('SFD_STEP',batch=batch,batches=len(sampler))
    scores = (totals/len(sampler)).cpu().numpy()
    n = max(1,round(c['sfd_percent']/100*512))
    channels = np.flatnonzero(scores<np.sort(scores)[n]).tolist()
    if len(channels)!=n:
        raise ValueError(('SFD strict-threshold tie/degeneracy, do not substitute channels',channels))
    channels = sorted(channels,key=lambda k:float(scores[k]))
    if rank==0:
        write(out/'SFD.json',dict(channels=channels,scores=scores.tolist(),train_sha256=c['input_hashes']['train'],
              classification_final_sha256=sha(out/'classification/last.pt'),signed_native_cosine_ranking=True,
              structural_parameter_removal=False))
    return channels


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['preflight','train','cuda-smoke'])
    p.add_argument('--config',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    c=json.loads(Path(a.config).read_text())
    out=Path(a.out)
    if a.command=='preflight':
        out.mkdir(exist_ok=False,parents=True)
        splits,receipt=read_inputs(c)
        decoded={}
        for split in ('train','val'):
            by={}
            for r in splits[split]:
                by.setdefault((r['label'],r['gender_race']),[]).append(r)
            selected=[r for v in by.values() for r in v[:2]]
            for r in selected:
                with Image.open(r['image']) as image:
                    image.convert('RGB').load()
            decoded[split]=len(selected)
        receipt.update(status='PASS',stratified_decoded_images=decoded,source_hashes=source_hashes(),
                       config_sha256=sha(a.config),manifest_hashes=c['input_hashes'])
        write(out/'PREFLIGHT.json',receipt)
        log('PREFLIGHT_PASS',**receipt)
        return
    if not torch.cuda.is_available() or int(os.environ.get('WORLD_SIZE','1'))!=2:
        raise RuntimeError('Requires two CUDA ranks; no CPU fallback')
    rank=int(os.environ['RANK'])
    local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.cuda.set_per_process_memory_fraction(c['cuda_memory_fraction'],local)
    torch.set_num_threads(c['threads_per_rank'])
    dist.init_process_group('nccl')
    device=torch.device('cuda',local)
    seed(c['seed'])
    splits,receipt=read_inputs(c,stat=False)
    if rank==0:
        out.mkdir(exist_ok=False,parents=True)
        write(out/'config.json',c)
        write(out/'START.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              source_hashes=source_hashes(),host=os.uname().nodename,pid=os.getpid(),world_size=2,
              cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_name=torch.cuda.get_device_name(local),
              input_receipt=receipt,smoke=a.command=='cuda-smoke',test_inference=False))
    dist.barrier()
    model=Detector(c['imagenet_weights'])
    if a.command=='cuda-smoke':
        # Real TRAIN images at full native global batches; disposable optimizer/model.
        before={k:v.clone() for k,v in model.state_dict().items()}
        model=train_stage(model,'classification',splits['train'],splits['val'],c,out,device,rank,2,True)
        if not any(not torch.equal(before[k],v.cpu()) for k,v in model.state_dict().items() if v.is_floating_point()):
            raise AssertionError('No pretraining updates')
        model.policy(True).train()
        frozen={k:v.cpu().clone() for k,v in model.state_dict().items() if 'last_linear' not in k}
        model=train_stage(model,'gda_corrected',splits['train'],splits['val'],c,out,device,rank,2,True)
        if any(not torch.equal(v,model.state_dict()[k].cpu()) for k,v in frozen.items()):
            raise AssertionError('Frozen encoder/BN changed')
        with torch.no_grad():
            x=torch.zeros(2,3,224,224,device=device)
            features=sfd_features(model.eval(),x)
            if features.shape[:2]!=(2,512) or not torch.isfinite(features).all():
                raise AssertionError('SFD feature shape/finite check')
        if rank==0:
            torch.save(model.state_dict(),out/'smoke_reload.pth')
            check=Detector().to(device)
            check.load_state_dict(torch.load(out/'smoke_reload.pth',map_location=device,weights_only=True))
            check.eval()
            with torch.no_grad():
                torch.testing.assert_close(check(x),model(x),rtol=0,atol=0)
            write(out/'CUDA_SMOKE_PASS.json',dict(status='PASS',real_train_only=True,
                  classification_global_images=128,gda_main_global_images=64,gda_aux_global_images=64,
                  frozen_encoder_and_bn_unchanged=True,checkpoint_roundtrip=True,sfd_feature_shape=list(features.shape),
                  source_hashes=source_hashes(),test_pixels_read=False))
            log('CUDA_SMOKE_PASS')
    else:
        model=train_stage(model,'classification',splits['train'],splits['val'],c,out,device,rank,2)
        channels=run_sfd(model,splits['train'],c,out,device,rank,2)
        mask_channels(model,channels)
        model.selected_channels=channels
        model=train_stage(model,'gda_corrected',splits['train'],splits['val'],c,out,device,rank,2)
        # Reload selected checkpoint, replay VAL and freeze threshold before any later TEST command.
        for stage in ('classification','gda_corrected'):
            ck=torch.load(out/stage/'best.pt',map_location='cpu',weights_only=False)
            selected=Detector()
            if ck['channels']:
                mask_channels(selected,ck['channels'])
            selected.load_state_dict(ck['model'],strict=True)
            selected=selected.to(device)
            z,cal,result=validation(selected,splits['val'],c,device,rank,2)
            if cal!=ck['calibration'] or result!=ck['validation']:
                raise AssertionError('Independent selected-checkpoint VAL replay mismatch')
            if rank==0:
                write(out/stage/'RUN_VERIFIED.json',dict(status='PASS',checkpoint_sha256=sha(out/stage/'best.pt'),
                      calibration=cal,validation=result,test_inference=False,selection='validation_auc_earliest_tie',
                      source_hashes=source_hashes()))
            del selected
        if rank==0:
            write(out/'DONE.json',dict(status='COMPLETED_VERIFIED',test_inference=False))
            log('TRAINING_COMPLETE',test_inference=False)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
