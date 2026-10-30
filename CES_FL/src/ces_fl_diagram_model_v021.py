#!/usr/bin/env python3
"""CES-FL diagram implementation v0.2.1 — explicit encoder policies; derived from v0.2 — standalone, exploratory, 30 October 2026.

Architecture matching image copy 46.png:
 A originals + demographic metadata; B one frozen ImageNet encoder;
 C global MLP and one shared 1x1 patch MLP;
 D learned sigmoid image/patch gates, positive global/local fusion;
 E binary logit and signed weighted local evidence map;
 F independently generated TRAIN edits + unchanged/resize controls, reviewed pairs,
   shared-weight forward passes and counterfactual losses. Qwen runs offline.

This is a NEW architecture, not the tested one-head v0.1, a Ding reproduction,
or a passed CESel-MA gate. No full training is started by running --help/smoke.
Learned gates are NOT proven reliability estimates; maps are NOT forgery masks.

Math: F=encoder(x), g=mean_hw(F), z_g=MLP(g), z_p=ConvMLP(F),
 r_g=sigmoid(G_g(g)), r_p=sigmoid(G_p(F)), a=softplus(fusion_raw),
 z=a_g*r_g*z_g + a_p*mean_hw(r_p*z_p) + b.
 Signed map M=a_p*r_p*z_p explains the LOCAL contribution only. The global
 contribution and fusion bias are exported separately; M.mean()+global+b=z.

Separate CES: BCE(z,y)+lambda_d*mean|asinh(z)-asinh(z_d)|
 +lambda_f*mean relu(tau-(2y-1)*(asinh(z)-asinh(z_f))).
 Coupled controlled ablation: BCE+lambda_s*mean_joint relu(tau+d-u),
 restricted to at most one demographic and one forensic pair per original.
 Optional map consistency is a NEW objective (default0), only for pairs flagged
 spatial_validated=1: lambda_map*mean|M-M_d|. It is NOT silently added to CES.

Dependencies: torch, torchvision (ResNet option), numpy, Pillow, scikit-learn.
Generation additionally needs diffusers/transformers/accelerate + local Qwen
weights. No package/model downloads occur automatically. Xception architecture
is embedded below from the project's vendored official implementation; no
other project Python file is needed. Public ImageNet weights are external data.

Manifest: image,label(0 real/1 fake),identity_ids,source_ids; optional gender,
race,age_bin,skin_bin,gender_race,ids8,family,cluster,age_estimate.
Pairs: pair_id,original_image,counterfactual_image,kind(demographic/forensic),
attribute(age/gender/skin/forensic),approved,identity_ids,source_ids;
forensic pairs also need forensic_validated=1; optional spatial_validated.
All edits/contributors must belong to TRAIN. No labels/approvals are fabricated.

Examples (use a new output directory):
 python ces_fl_diagram_model.py smoke --device cpu --out /tmp/diagram_cpu
 python ces_fl_diagram_model.py train --train TRAIN.csv --val VAL.csv \
   --pairs REVIEWED.csv --mode invariance --out RUN --device cuda --seed 17
 python ces_fl_diagram_model.py evaluate --checkpoint RUN/best.pt \
   --manifest TEST.csv --out TEST_REPORT --device cuda
 python ces_fl_diagram_model.py export --checkpoint RUN/best.pt \
   --images IMAGE.png --out MAPS --device cuda
 python ces_fl_diagram_model.py generate --train TRAIN.csv --out EDITS \
   --max-originals 6 --plan-only

Use natural and balanced BCE controls. Freeze selection/hyperparameters before
test evaluation. Default numerical settings are unoptimized engineering values.
Human visual approval does not establish identity/authenticity/forensic validity.
"""
import argparse
from contextlib import nullcontext
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import time
import sys
import tempfile

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler

VERSION='CESFL_DIAGRAM_v0.2.1'
LEGACY_VERSION='CESFL_DIAGRAM_v0.2'
DEFAULT_XCEPTION='/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/data/xception-b5690688.pth'
DEFAULT_QWEN='/home/sachin.chaudhary/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9'
AXES=('gender','race','age_bin','skin_bin','gender_race','ids8')

# Embedded Xception classes reused from the project's vendored Ding utils/xception.py.
# Provenance: ces_fl_pipeline/vendor/ding_xception.py (classes only, registry removed).
# Source paths are attribution only; no import of those project files occurs.
# ---- BEGIN VERBATIM 1 (Ding_mod/utils/xception.py lines 25-93)
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:   # whether the number of filters grows first
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

# ---- END VERBATIM 1


# ---- BEGIN VERBATIM 2 (Ding_mod/utils/xception.py lines 102-289; decorator line 101 dropped)
class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, xception_config):
        """ Constructor
        Args:
            xception_config: configuration file with the dict format
        """
        super(Xception, self).__init__()
        self.num_classes = xception_config["num_classes"]
        self.mode = xception_config["mode"]
        inc = xception_config["inc"]
        dropout = xception_config["dropout"]

        # Entry flow
        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # middle flow
        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        # if self.mode == 'adjust_channel':
        #     self.adjust_channel = nn.Sequential(
        #         nn.Conv2d(2048, 512, 1, 1),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True),
        #     )
        #     self.last_linear = nn.Linear(512, self.num_classes)
        # else:
        #     self.last_linear = nn.Linear(2048, self.num_classes)

        # dag
        self.last_linear = nn.Linear(512, self.num_classes)
        if dropout:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(2048, self.num_classes)
            )

        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )




        # if dropout:
        #     self.last_linear = nn.Sequential(
        #         nn.Dropout(p=dropout),
        #         nn.Linear(2048, self.num_classes)
        #     )

    def fea_part1_0(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def fea_part1_1(self, x):

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def fea_part2(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def fea_part3(self, x):
        if self.mode == "shallow_xception":
            return x
        else:
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
        return x

    def fea_part4(self, x):
        if self.mode == "shallow_xception":
            x = self.block12(x)
        else:
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            # print(x.shape)
            x = self.block12(x)
            # print(x.shape, 12)
        return x

    def fea_part5(self, x):
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x

    def features(self, input):
        x = self.fea_part1(input)

        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)

        x = self.fea_part5(x)

        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)

        return x

    def classifier(self, features):
        # x = self.relu(features) #([128, 512, 7, 7])
        x = F.adaptive_avg_pool2d(features, (1, 1)) #([128, 512, 1, 1])
        x = x.view(x.size(0), -1) #([128, 512])
        out = self.last_linear(x)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out, x

# ---- END VERBATIM 2


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def empty_out(path):
    p=Path(path)
    if p.exists() and any(p.iterdir()):raise ValueError(f'Preserve prior outputs: {p}')
    p.mkdir(parents=True,exist_ok=True);return p


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def flag(value):
    s=str(value).strip().lower()
    if s not in ('','0','1','false','true','no','yes'):raise ValueError(f'Invalid boolean flag: {value}')
    return s in ('1','true','yes')


class DiagramCES(nn.Module):
    def __init__(self,backbone='xception',weights=DEFAULT_XCEPTION,hidden=128,gate_hidden=64,pretrained=True,encoder_policy='frozen',bn_policy='frozen',head_seed=None):
        super().__init__();self.backbone=backbone;self.hidden=hidden;self.gate_hidden=gate_hidden
        if backbone=='xception':
            self.encoder=Xception({'mode':'original','num_classes':2,'inc':3,'dropout':False})
            if pretrained:
                if not Path(weights).is_file():raise FileNotFoundError(weights)
                sd=torch.load(weights,map_location='cpu',weights_only=True)
                sd={k:(v.unsqueeze(-1).unsqueeze(-1) if 'pointwise' in k and v.ndim==2 else v) for k,v in sd.items() if 'fc' not in k}
                result=self.encoder.load_state_dict(sd,strict=False)
                if result.unexpected_keys:raise ValueError(result.unexpected_keys)
                if any(not k.startswith(('adjust_channel.','last_linear.')) for k in result.missing_keys):raise ValueError(result.missing_keys)
            # Use pretrained2048-channel features, removing the unused RANDOM Ding512 adaptation.
            # This explicitly differs from v0.1; it avoids freezing random untrained adaptation weights.
            self.encoder.adjust_channel=nn.Identity();self.encoder.last_linear=nn.Identity();channels=2048
        elif backbone=='resnet50':
            from torchvision.models import resnet50
            m=resnet50(weights=None)
            if pretrained:
                if not weights or not Path(weights).is_file():raise FileNotFoundError('Supply a local ResNet50 ImageNet checkpoint via --weights')
                m.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
            self.encoder=nn.Sequential(*list(m.children())[:-2]);channels=2048
        else:raise ValueError('Supported encoders: xception,resnet50')
        self.encoder.requires_grad_(False)
        self.global_head=nn.Sequential(nn.Linear(channels,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.patch_head=nn.Sequential(nn.Conv2d(channels,hidden,1),nn.ReLU(),nn.Conv2d(hidden,1,1))
        self.image_gate=nn.Sequential(nn.Linear(channels,gate_hidden),nn.ReLU(),nn.Linear(gate_hidden,1),nn.Sigmoid())
        self.patch_gate=nn.Sequential(nn.Conv2d(channels,gate_hidden,1),nn.ReLU(),nn.Conv2d(gate_hidden,1,1),nn.Sigmoid())
        self.fusion_raw=nn.Parameter(torch.full((2,),math.log(math.expm1(1.))))
        self.fusion_bias=nn.Parameter(torch.zeros(()))
        if head_seed is not None:self.reset_prediction_modules(head_seed)
        self.set_encoder_policy(encoder_policy,bn_policy)

    def reset_prediction_modules(self,seed):
        # A/B/C reset only prediction modules; preserve all encoder parameters/buffers.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for name in ('global_head','patch_head','image_gate','patch_gate'):
                for module in getattr(self,name).modules():
                    if isinstance(module,(nn.Linear,nn.Conv2d)):module.reset_parameters()
            with torch.no_grad():
                self.fusion_raw.fill_(math.log(math.expm1(1.)));self.fusion_bias.zero_()

    def set_encoder_policy(self,encoder_policy,bn_policy):
        if encoder_policy not in ('frozen','trainable') or bn_policy not in ('frozen','update'):
            raise ValueError('Unknown encoder/BatchNorm policy')
        if encoder_policy=='frozen' and bn_policy!='frozen':raise ValueError('Frozen encoder requires frozen BN')
        self.encoder_policy,self.bn_policy=encoder_policy,bn_policy
        self.encoder.requires_grad_(encoder_policy=='trainable')
        self.train(self.training)

    def train(self,mode=True):
        super().train(mode)
        if self.encoder_policy=='frozen':self.encoder.eval()
        elif self.bn_policy=='frozen':
            for module in self.encoder.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self

    def forward(self,x):
        with (torch.no_grad() if self.encoder_policy=='frozen' else nullcontext()):
            spatial=self.encoder.features(x) if self.backbone=='xception' else self.encoder(x)
        global_features=spatial.mean((2,3))
        zg=self.global_head(global_features).squeeze(1);zp=self.patch_head(spatial).squeeze(1)
        rg=self.image_gate(global_features).squeeze(1);rp=self.patch_gate(spatial).squeeze(1)
        weights=F.softplus(self.fusion_raw)
        global_contribution=weights[0]*rg*zg;local_map=weights[1]*rp*zp
        local_contribution=local_map.mean((1,2));z=global_contribution+local_contribution+self.fusion_bias
        return {'logit':z,'global_score':zg,'patch_score':zp,'image_gate':rg,'patch_gate':rp,
            'local_map':local_map,'global_contribution':global_contribution,'local_contribution':local_contribution,
            'fusion_bias':self.fusion_bias,'fusion_weights':weights}

    def counts(self):return {'total':sum(p.numel() for p in self.parameters()),'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad)}


def ces_loss(original,y,demo=None,di=None,forensic=None,fi=None,*,mode='combined',selectivity='separate',
             lambda_d=1.,lambda_f=1.,lambda_s=1.,tau=.5,forensic_scope='fake',lambda_map=0.,spatial_valid=None):
    if mode not in ('bce','invariance','sensitivity','combined'):raise ValueError(mode)
    if selectivity not in ('separate','coupled') or (selectivity=='coupled' and mode!='combined'):raise ValueError('Invalid selectivity/mode')
    z=original['logit'];y=y.to(z.device).float();zero=z.sum()*0
    if z.ndim!=1 or z.shape!=y.shape or not ((y==0)|(y==1)).all():raise ValueError('Original shape/labels invalid')
    def prepare(value,index):
        if value is None:return z.new_empty(0),torch.empty(0,dtype=torch.long,device=z.device)
        v=value['logit'];ix=index.to(z.device,dtype=torch.long)
        if v.ndim!=1 or len(v)!=len(ix) or (len(ix) and (ix.min()<0 or ix.max()>=len(z))):raise ValueError('Pair alignment invalid')
        return v,ix
    zd,di=prepare(demo,di);zf,fi=prepare(forensic,fi)
    if forensic_scope=='fake':keep=y[fi]==1;zf,fi=zf[keep],fi[keep]
    elif forensic_scope!='all':raise ValueError(forensic_scope)
    d=(torch.asinh(z[di])-torch.asinh(zd)).abs() if mode in ('invariance','combined') else z.new_empty(0)
    u=(2*y[fi]-1)*(torch.asinh(z[fi])-torch.asinh(zf)) if mode in ('sensitivity','combined') else z.new_empty(0)
    cls=F.binary_cross_entropy_with_logits(z,y);inv=d.mean() if d.numel() else zero
    sens=F.relu(tau-u).mean() if u.numel() else zero;coupled=zero;joint_count=0
    if selectivity=='coupled':
        dc=torch.bincount(di,minlength=len(z));fc=torch.bincount(fi,minlength=len(z))
        if (dc>1).any() or (fc>1).any():raise ValueError('Coupled needs at most one pair of each kind per original')
        joint=(dc==1)&(fc==1);joint_count=int(joint.sum())
        if joint_count:
            dv=z.new_zeros(len(z)).index_add(0,di,d);uv=z.new_zeros(len(z)).index_add(0,fi,u)
            coupled=F.relu(tau+dv[joint]-uv[joint]).mean()
    map_loss=zero;map_count=0
    if lambda_map and d.numel():
        if spatial_valid is None:raise ValueError('Map consistency needs explicit spatial_validated flags')
        keep=spatial_valid.to(z.device,dtype=torch.bool)
        if keep.shape!=di.shape:raise ValueError('Spatial flags misaligned')
        map_count=int(keep.sum())
        if map_count:
            if original['local_map'].shape[1:]!=demo['local_map'].shape[1:]:raise ValueError('Spatial grids differ')
            map_loss=(original['local_map'][di[keep]]-demo['local_map'][keep]).abs().mean()
    total=cls+(lambda_s*coupled if selectivity=='coupled' else lambda_d*inv+lambda_f*sens)+lambda_map*map_loss
    return {'total':total,'classification':cls,'invariance':inv,'sensitivity':sens,'coupled':coupled,'map_consistency':map_loss,
        'demo_count':d.numel(),'forensic_count':u.numel(),'joint_count':joint_count,'map_count':map_count}


def read_rows(path):
    path=Path(path)
    with path.open(newline='') as f:
        r=csv.DictReader(f)
        if not {'image','label','identity_ids','source_ids'}.issubset(r.fieldnames or []):raise ValueError('Missing manifest image/label/identity_ids/source_ids')
        rows=list(r)
    images=set()
    for row in rows:
        p=Path(row['image']);p=(path.parent/p).resolve() if not p.is_absolute() else p.resolve();row['image']=str(p)
        if row['label'] not in ('0','1') or not p.is_file():raise ValueError(f'Invalid label/file: {row}')
        row['label']=int(row['label'])
        if str(p) in images:raise ValueError('Duplicate original image')
        images.add(str(p))
        if not row['identity_ids'].strip() or not row['source_ids'].strip():raise ValueError('Missing contributor provenance')
        for axis in AXES:row[axis]=row.get(axis) or 'unknown'
        row['cluster']=row.get('cluster') or row['identity_ids'].split(';')[0];row['family']=row.get('family') or 'unknown'
    if {r['label'] for r in rows}!={0,1}:raise ValueError('Both classes required')
    return rows


def check_splits(train,val,declared_file=None):
    declared=json.loads(Path(declared_file).read_text())['declared_overlap_values'] if declared_file else {}
    if {r['image'] for r in train}&{r['image'] for r in val}:raise ValueError('Image-path leakage')
    observed={}
    for field in ('identity_ids','source_ids'):
        a={v for r in train for v in r[field].split(';')};b={v for r in val for v in r[field].split(';')}
        overlap=sorted(a&b);key=field+':train:val'
        if overlap!=declared.get(key,[]):raise ValueError(f'Unrecorded {field} overlap or changed benchmark inventory')
        observed[field]=len(overlap)
    return {'overlap_counts':observed,'exact_declared_benchmark_exception':bool(declared),'person_ground_truth_verified':False}


def read_pairs(path,rows,attributes):
    if not path:return {},{'approved':0,'by_attribute':{}}
    with Path(path).open() as f:pairs=list(csv.DictReader(f))
    index={r['image']:r for r in rows};out={};ids=set();counts={}
    for r in pairs:
        if r['pair_id'] in ids:raise ValueError('Duplicate pair_id')
        ids.add(r['pair_id']);source=str(Path(r['original_image']).resolve())
        if source not in index:raise ValueError('Counterfactual source is outside TRAIN')
        if not flag(r.get('approved',0)):continue
        kind=r.get('kind','demographic');attr=r.get('attribute','')
        if kind not in ('demographic','forensic'):raise ValueError('Invalid pair kind')
        if kind=='demographic' and attr not in attributes:continue
        if kind=='forensic':
            attr='forensic'
            if not flag(r.get('forensic_validated',0)):raise ValueError('Forensic intervention needs independent validation')
        cf=Path(r['counterfactual_image']).resolve()
        if not cf.is_file():raise FileNotFoundError(cf)
        for field in ('identity_ids','source_ids'):
            values=set(r.get(field,'').split(';'))-{''}
            if not values or not values.issubset(set(index[source][field].split(';'))):raise ValueError('Undeclared CF contributor')
        r={**r,'counterfactual_image':str(cf),'spatial_validated':flag(r.get('spatial_validated',0))}
        out.setdefault(source,{}).setdefault(attr,[]).append(r);counts[attr]=counts.get(attr,0)+1
    return out,{'approved':sum(counts.values()),'by_attribute':counts}


class PairImages(Dataset):
    def __init__(self,rows,pairs=None,seed=17,epoch=0,mode='bce',size=224,backbone='xception'):
        self.rows,self.pairs,self.seed,self.epoch,self.mode,self.size,self.backbone=rows,pairs or {},seed,epoch,mode,size,backbone
    def __len__(self):return len(self.rows)
    def load(self,path):
        with Image.open(path) as im:a=np.asarray(im.convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR),dtype=np.float32).copy()/255.
        x=torch.from_numpy(a).permute(2,0,1)
        if self.backbone=='xception':return (x-.5)/.5
        return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    def __getitem__(self,i):
        r=self.rows[i];demo=[];forensic=[]
        for attr,choices in sorted(self.pairs.get(r['image'],{}).items()):
            if attr=='forensic' and self.mode not in ('sensitivity','combined'):continue
            if attr!='forensic' and self.mode not in ('invariance','combined'):continue
            offset=int(hashlib.sha256(f'{self.seed}:{r["image"]}:{attr}'.encode()).hexdigest()[:16],16)
            p=choices[(offset+self.epoch)%len(choices)]
            (forensic if attr=='forensic' else demo).append((self.load(p['counterfactual_image']),p['pair_id'],attr,p['spatial_validated']))
        return self.load(r['image']),r['label'],demo,forensic,i


def collate(items):
    result={'x':torch.stack([r[0] for r in items]),'y':torch.tensor([r[1] for r in items],dtype=torch.float32),'rows':[r[4] for r in items]}
    for key,pos in [('demo',2),('forensic',3)]:
        pairs=[(j,p) for j,r in enumerate(items) for p in r[pos]]
        result[key]=torch.stack([p[0] for _,p in pairs]) if pairs else None
        result[key+'_index']=torch.tensor([j for j,_ in pairs],dtype=torch.long)
        result[key+'_ids']=[p[1] for _,p in pairs];result[key+'_attributes']=[p[2] for _,p in pairs]
        result[key+'_spatial']=torch.tensor([p[3] for _,p in pairs],dtype=torch.bool)
    return result


def calibrate(real,target=.1):
    z=np.asarray(real,dtype=np.float64)
    if z.ndim!=1 or not len(z) or not np.isfinite(z).all() or not 0<=target<1:raise ValueError('Invalid calibration')
    f=Fraction(str(target))*len(z);budget=f.numerator//f.denominator
    threshold=float(np.nextafter(np.sort(z)[::-1][budget],np.inf))
    if not math.isfinite(threshold):raise ValueError('Cannot represent finite threshold')
    fp=int((z>=threshold).sum())
    return {'threshold_logit':threshold,'rule':'validation_real_empirical_budget_v1','requested_fpr':target,'n_validation_real':len(z),
        'allowed_fp':budget,'achieved_fp':fp,'achieved_fpr':fp/len(z),'test_fpr_guaranteed':False}


def metrics(rows,logits,threshold,min_support=20):
    from sklearn.metrics import roc_auc_score
    z=np.asarray(logits);y=np.asarray([r['label'] for r in rows]);pred=z>=threshold
    if len(z)!=len(rows) or not np.isfinite(z).all():raise ValueError('Invalid logits')
    def one(mask):
        yy,zz,pp=y[mask],z[mask],pred[mask];real,fake=yy==0,yy==1
        return {'n':int(len(yy)),'n_real':int(real.sum()),'n_fake':int(fake.sum()),
            'auc':float(roc_auc_score(yy,zz)) if real.any() and fake.any() else None,
            'fpr':float(pp[real].mean()) if real.any() else None,'fnr':float((~pp[fake]).mean()) if fake.any() else None,
            'positive_rate':float(pp.mean()) if len(pp) else None}
    overall=one(np.ones(len(y),bool));axes={}
    for axis in AXES:
        labels=np.asarray([r.get(axis,'unknown') for r in rows]);groups={str(g):one(labels==g) for g in np.unique(labels)}
        good={g:r for g,r in groups.items() if g!='unknown' and min(r['n_real'],r['n_fake'])>=min_support}
        axes[axis]={'groups':groups,'eligible_groups':list(good),'min_support_per_class':min_support,
            'worst_group_auc':min(r['auc'] for r in good.values()) if good else None}
        for key in ('auc','fpr','fnr'):
            values=[r[key] for r in good.values()];axes[axis][key+'_gap']=max(values)-min(values) if len(values)>1 else None
        axes[axis]['ding_style_fractions']={'F_FPR':sum(abs(r['fpr']-overall['fpr']) for r in good.values()),
            'F_DP':max(r['positive_rate'] for r in good.values())-min(r['positive_rate'] for r in good.values()),
            'es_AUC':overall['auc']/(1+sum(abs(r['auc']-overall['auc']) for r in good.values()))} if len(good)>1 else None
    families=np.asarray([r.get('family','unknown') for r in rows]);by_family={}
    for family in np.unique(families):
        mask=families==family if family=='real' else (y==0)|((y==1)&(families==family))
        by_family[str(family)]=one(mask)
    return {'units':'fractions','threshold_logit':float(threshold),'overall':overall,'axes':axes,'by_family':by_family}


@torch.inference_mode()
def predict(model,rows,device,batch_size=16,workers=0,size=224):
    model.eval();out=[]
    for b in DataLoader(PairImages(rows,size=size,backbone=model.backbone),batch_size=batch_size,collate_fn=collate,num_workers=workers):
        out.extend(model(b['x'].to(device))['logit'].cpu().tolist())
    return np.asarray(out)


def slice_output(output,start,end):return {k:output[k][start:end] for k in ('logit','local_map')}


def build_optimizer(model,c):
    encoder=[p for p in model.encoder.parameters() if p.requires_grad]
    other=[p for name,p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
    groups=[]
    if encoder:groups.append({'params':encoder,'lr':c.get('encoder_lr',c['lr']),'name':'encoder'})
    if other:groups.append({'params':other,'lr':c['lr'],'name':'heads_gates_fusion'})
    if not groups:raise ValueError('No trainable parameters')
    if not all(math.isfinite(g['lr']) and g['lr']>0 for g in groups):raise ValueError('Invalid learning rate')
    return torch.optim.AdamW(groups,lr=c['lr'],weight_decay=c['weight_decay'])


def matched_resize_image(image,target_area):
    """Qwen geometric input resize and BOX canonicalization; no latent reconstruction."""
    image=image.convert('RGB');w,h=image.size;ratio=w/h
    width=round(math.sqrt(target_area*ratio)/32)*32
    height=round(math.sqrt(target_area/ratio)/32)*32
    if min(width,height)<32:raise ValueError('Unsupported extreme aspect ratio')
    return image.resize((width,height),Image.Resampling.LANCZOS).resize((w,h),Image.Resampling.BOX)


def read_controls(path,train,pairs):
    sources=json.loads(Path(path).read_text());index={r['image']:r for r in train};seen=set()
    for r in sources:
        source=r['original_image']
        if source not in index or source in seen:raise ValueError('Invalid/duplicate control source')
        seen.add(source)
        if sha(source)!=r['original_sha256']:raise ValueError('Original changed')
        for field in ('nochange_control','resize_control','condition_resize_control'):
            if not r.get(field) or sha(r[field])!=r[field+'_sha256']:raise ValueError('Control missing/changed: '+field)
        edits={p['pair_id']:p for pp in pairs.get(source,{}).values() for p in pp}
        recorded={e['pair_id']:e for e in r['edits']}
        for pair_id,pair in edits.items():
            if pair_id not in recorded:raise ValueError('Approved pair missing associated controls')
            entry=recorded[pair_id]
            if entry['counterfactual_image']!=pair['counterfactual_image'] or not entry['independent_from_original']:
                raise ValueError('Mismatched or chained intervention')
            if sha(entry['counterfactual_image'])!=entry['sha256']:raise ValueError('Edit changed')
    if set(pairs)-seen:raise ValueError('Approved sources missing controls')
    return sources


@torch.inference_mode()
def diagnose_controls(checkpoint,manifest,out,device):
    model,ck=load_model(checkpoint,device);out=empty_out(out)
    sources=json.loads(Path(manifest).read_text());result=[]
    loader=PairImages([],size=ck['config']['size'],backbone=model.backbone)
    for source in sources:
        paths=[('original',source['original_image'])]+[(k,source[k]) for k in
            ('nochange_control','resize_control','condition_resize_control')]
        paths += [(e['pair_id'],e['counterfactual_image']) for e in source['edits']]
        scores=[]
        for _,path in paths:
            scores.append(float(model(loader.load(path)[None].to(device))['logit'][0]))
        with Image.open(source['original_image']) as im:original=np.asarray(im.convert('RGB'),dtype=float)
        for (condition,path),z in zip(paths,scores):
            with Image.open(path) as im:edited=np.asarray(im.convert('RGB').resize((original.shape[1],original.shape[0]),Image.Resampling.BOX),dtype=float)
            result.append({'original_image':source['original_image'],'condition':condition,'image':path,
                'sha256':sha(path),'logit':z,'logit_delta_vs_original':z-scores[0],
                'asinh_delta_vs_original':float(np.arcsinh(z)-np.arcsinh(scores[0])),
                'probability_delta_vs_original':float(torch.sigmoid(torch.tensor(z))-torch.sigmoid(torch.tensor(scores[0]))),
                'pixel_mae_0_255':float(abs(edited-original).mean())})
    write(out/'diagnostics.json',{'checkpoint_sha256':sha(checkpoint),'controls_sha256':sha(manifest),
        'rows':result,'interpretation':'Diagnostics only; no new control loss, validity approval or fairness claim. Gates are learned weights; maps are signed local contributions.'})
    return result


def import_encoder(model,c):
    path=c['encoder_checkpoint'];expected=c.get('encoder_checkpoint_sha256')
    if not expected or sha(path)!=expected:raise ValueError('Encoder checkpoint SHA256 missing/mismatched')
    ck=torch.load(path,map_location='cpu',weights_only=True);pc=ck['config']
    if ck.get('format')!=VERSION or pc.get('stage')!='pretrain' or pc.get('encoder_policy')!='trainable' or pc['mode']!='bce':
        raise ValueError('Need a verified original-TRAIN BCE encoder-pretraining checkpoint')
    for key in ('backbone','size'):
        if pc[key]!=c[key]:raise ValueError('Encoder/preprocessing mismatch: '+key)
    if ck['training_inputs']!={'train_sha256':sha(c['train']),'val_sha256':sha(c['val'])}:
        raise ValueError('Pretraining/downstream split hashes differ')
    model.encoder.load_state_dict({k[len('encoder.'):]:v for k,v in ck['model'].items() if k.startswith('encoder.')},strict=True)
    return {'path':str(path),'sha256':expected,'epoch':ck['epoch'],'imported':'encoder only; fresh identical downstream heads/gates/fusion'}


def train_run(c,out,device):
    if c['mode'] not in ('bce','invariance','sensitivity','combined'):raise ValueError('Invalid mode')
    if c.get('selectivity','separate')=='coupled' and len(c['attributes'])>1:raise ValueError('Coupled ablation requires one demographic attribute')
    for key in ('lambda_d','lambda_f','lambda_s','lambda_map','tau','lr','weight_decay'):
        if not math.isfinite(c[key]) or c[key]<0:raise ValueError(f'Invalid {key}')
    if c['epochs']<1 or c['batch_size']<1 or c['workers']<0:raise ValueError('Invalid execution settings')
    rows=read_rows(c['train']);val=read_rows(c['val']);provenance=check_splits(rows,val,c.get('protocol_overlap_file'))
    pairs,pstat=read_pairs(c.get('pairs'),rows,c['attributes'])
    controls=read_controls(c['controls_manifest'],rows,pairs) if c.get('controls_manifest') else None
    if c.get('require_controls') and controls is None:raise ValueError('Associated controls required')
    val_paths={r['image'] for r in val}
    for groups in pairs.values():
        for pp in groups.values():
            if any(p['counterfactual_image'] in val_paths for p in pp):raise ValueError('CF overlaps validation')
    if controls:
        for source in controls:
            if any(source[k] in val_paths for k in ('nochange_control','resize_control','condition_resize_control') if source.get(k)):
                raise ValueError('Control overlaps validation')
    if c['mode'] in ('invariance','combined'):
        missing=[a for a in c['attributes'] if not pstat['by_attribute'].get(a)]
        if not c['attributes'] or missing:raise ValueError(f'Approved demographic pairs missing: {missing}')
    if c['mode'] in ('sensitivity','combined') and not pstat['by_attribute'].get('forensic'):raise ValueError('No independently validated forensic pairs')
    if c['lambda_map'] and not any(p['spatial_validated'] for g in pairs.values() for a,pp in g.items() if a!='forensic' for p in pp):raise ValueError('No spatially validated pairs for map consistency')
    c={**c,'encoder_policy':c.get('encoder_policy','frozen'),'bn_policy':c.get('bn_policy','frozen'),
        'head_seed':c['seed'] if c.get('head_seed') is None else c['head_seed'],'stage':c.get('stage','downstream')}
    if c['stage'] not in ('pretrain','downstream'):raise ValueError('Invalid training stage')
    if c['stage']=='pretrain':
        if c['mode']!='bce' or c.get('pairs') or c['encoder_policy']!='trainable':
            raise ValueError('Pretraining requires trainable encoder, original TRAIN images and BCE only')
    elif c['encoder_policy']!='frozen':raise ValueError('Controlled downstream experiment requires frozen encoder')
    if c.get('encoder_checkpoint') and c['stage']!='downstream':raise ValueError('Encoder import is downstream only')
    out=empty_out(out);seed_all(c['seed']);torch.set_num_threads(4)
    model=DiagramCES(c['backbone'],c['weights'],c['hidden'],c['gate_hidden'],
        encoder_policy=c['encoder_policy'],bn_policy=c['bn_policy'],head_seed=c['head_seed']).to(device)
    encoder_import=import_encoder(model,c) if c.get('encoder_checkpoint') else None
    opt=build_optimizer(model,c)
    write(out/'config.json',c);write(out/'provenance.json',{'version':VERSION,'script':str(Path(__file__).resolve()),'script_sha256':sha(__file__),
        'input_sha256':{str(c[k]):sha(c[k]) for k in ('train','val','pairs','protocol_overlap_file') if c.get(k)},'weights_sha256':sha(c['weights']),
        'parameters':model.counts(),'splits':provenance,'approved_pairs':pstat,'test_read_during_training':False,'architecture':'global+patch heads, learned gates, positive fusion','encoder_policy':c['encoder_policy'],
        'bn_policy':c['bn_policy'],'encoder_import':encoder_import,'controls_manifest_sha256':sha(c['controls_manifest']) if controls else None,
        'optimizer_groups':[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups]})
    best=-math.inf;all_pair_draws={}
    for epoch in range(c['epochs']):
        model.train();dataset=PairImages(rows,pairs,c['seed'],epoch,c['mode'],c['size'],c['backbone'])
        sampler=None
        if c['sampler']=='balanced':
            counts={}
            for r in rows:key=(r['label'],r[c['balance_axis']]);counts[key]=counts.get(key,0)+1
            weights=[1./counts[(r['label'],r[c['balance_axis']])] for r in rows]
            sampler=WeightedRandomSampler(weights,len(rows),replacement=True,generator=torch.Generator().manual_seed(c['seed']+epoch))
        loader=DataLoader(dataset,batch_size=c['batch_size'],sampler=sampler,shuffle=sampler is None,num_workers=c['workers'],collate_fn=collate,
            generator=torch.Generator().manual_seed(c['seed']+epoch))
        totals={k:0. for k in ('classification','invariance','sensitivity','coupled','map_consistency')};den={k:0 for k in totals};pair_draws={};steps=0
        for b in loader:
            n=len(b['y']);nd=0 if b['demo'] is None else len(b['demo']);nf=0 if b['forensic'] is None else len(b['forensic'])
            parts=[b['x']]+([b['demo']] if nd else [])+([b['forensic']] if nf else [])
            output=model(torch.cat(parts).to(device))
            loss=ces_loss(slice_output(output,0,n),b['y'].to(device),slice_output(output,n,n+nd),b['demo_index'],
                slice_output(output,n+nd,n+nd+nf),b['forensic_index'],mode=c['mode'],selectivity=c['selectivity'],lambda_d=c['lambda_d'],
                lambda_f=c['lambda_f'],lambda_s=c['lambda_s'],tau=c['tau'],lambda_map=c['lambda_map'],spatial_valid=b['demo_spatial'],forensic_scope=c['forensic_scope'])
            if not torch.isfinite(loss['total']):raise RuntimeError('Nonfinite loss')
            opt.zero_grad(set_to_none=True);loss['total'].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradient')
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.);opt.step();steps+=1
            for key,count in [('classification',n),('invariance',loss['demo_count']),('sensitivity',loss['forensic_count']),('coupled',loss['joint_count']),('map_consistency',loss['map_count'])]:
                totals[key]+=float(loss[key].detach())*count;den[key]+=count
            for pair_id in b['demo_ids']+b['forensic_ids']:pair_draws[pair_id]=pair_draws.get(pair_id,0)+1
        z=predict(model,val,device,c['batch_size'],c['workers'],c['size']);y=np.asarray([r['label'] for r in val])
        calibration=calibrate(z[y==0],c['target_fpr']);report=metrics(val,z,calibration['threshold_logit'],c['min_support'])
        record={'epoch':epoch+1,'steps':steps,'validation':report,'calibration':calibration,'selection':'validation_auc_only',
            'losses':{k:totals[k]/den[k] if den[k] else None for k in totals},'eligible_counts':den,'pair_draws':pair_draws}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps({'epoch':epoch+1,'steps':steps,'val_auc':report['overall']['auc'],'losses':record['losses']}),flush=True)
        ck={'format':VERSION,'model':model.state_dict(),'config':c,'epoch':epoch+1,
            'calibration':calibration,'validation':report,'optimizer':opt.state_dict(),
            'training_inputs':{'train_sha256':sha(c['train']),'val_sha256':sha(c['val'])},'encoder_import':encoder_import}
        if report['overall']['auc']>best:
            best=report['overall']['auc'];torch.save(ck,out/'best.pt')
        torch.save(ck,out/'last.pt')
        for pair_id,count in pair_draws.items():all_pair_draws[pair_id]=all_pair_draws.get(pair_id,0)+count
        write(out/'pair_usage.json',{'unique_pairs_used':len(all_pair_draws),'pair_draws':all_pair_draws})
    (out/'DONE').write_text('Training completed; test not read\n');return out/'best.pt'


def load_model(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('format') not in (VERSION,LEGACY_VERSION):raise ValueError('Checkpoint is not this NEW diagram architecture; do not reuse v0.1 head weights')
    c=ck['config'];m=DiagramCES(c['backbone'],c['weights'],c['hidden'],c['gate_hidden'],pretrained=False,encoder_policy=c.get('encoder_policy','frozen'),bn_policy=c.get('bn_policy','frozen'))
    m.load_state_dict(ck['model'],strict=True);return m.to(device).eval(),ck


def evaluate_run(checkpoint,manifest,out,device):
    model,ck=load_model(checkpoint,device);threshold=ck['calibration']['threshold_logit']
    # The checkpoint already fixes threshold before any evaluation scores or labels are used.
    rows=read_rows(manifest);z=predict(model,rows,device,ck['config']['batch_size'],ck['config']['workers'],ck['config']['size']);out=empty_out(out)
    report={'checkpoint_sha256':sha(checkpoint),'manifest_sha256':sha(manifest),'threshold_source':'selected checkpoint validation only',
        'calibration':ck['calibration'],'calibrated':metrics(rows,z,threshold,ck['config']['min_support']),
        'fixed_probability_0_5':metrics(rows,z,0.,ck['config']['min_support'])}
    write(out/'metrics.json',report)
    with (out/'predictions.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['image','label','logit','prediction']);w.writerows((r['image'],r['label'],float(s),int(s>=threshold)) for r,s in zip(rows,z))
    print('EVALUATED',len(rows),'frozen threshold',threshold);return report


def diverging(a,scale):
    t=np.clip(np.asarray(a,dtype=float)/max(scale,1e-12),-1,1);out=np.ones((*t.shape,3),float)
    out[...,0]=1-np.maximum(-t,0);out[...,1]=1-abs(t);out[...,2]=1-np.maximum(t,0)
    return (out*255).astype(np.uint8)


@torch.inference_mode()
def export_maps(checkpoint,images,out,device):
    model,ck=load_model(checkpoint,device);out=empty_out(out);rows=[]
    for i,path in enumerate(images):
        p=Path(path).resolve();x=PairImages([],size=ck['config']['size'],backbone=model.backbone).load(p).unsqueeze(0).to(device);r=model(x)
        with Image.open(p) as im:im=im.convert('RGB');size=im.size
        raw=r['local_map'][0].cpu().numpy();up=F.interpolate(r['local_map'][:,None],size=size[::-1],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        scale=max(float(np.max(abs(up))),1e-12);heat=Image.fromarray(diverging(up,scale));name=f'{i:04d}_{p.stem}'
        Image.blend(im,heat,.45).save(out/f'{name}_local_overlay.png');heat.save(out/f'{name}_local_signed.png')
        gate=F.interpolate(r['patch_gate'][:,None],size=size[::-1],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        Image.fromarray(np.rint(gate*255).astype(np.uint8)).save(out/f'{name}_patch_gate.png')
        np.savez_compressed(out/f'{name}_raw.npz',local_map=raw,upsampled_local_map=up,patch_gate=r['patch_gate'][0].cpu().numpy())
        row={'image':str(p),'source_sha256':sha(p),'logit':float(r['logit'][0]),'global_contribution':float(r['global_contribution'][0]),
            'local_contribution':float(r['local_contribution'][0]),'fusion_bias':float(r['fusion_bias']),'image_gate':float(r['image_gate'][0]),
            'fusion_weights':r['fusion_weights'].cpu().tolist(),'display_abs_scale':scale,'map_is_ground_truth':False}
        assert abs(row['logit']-(row['global_contribution']+float(raw.mean())+row['fusion_bias']))<1e-5;rows.append(row)
    write(out/'scores.json',rows)
    (out/'README.md').write_text('Signed LOCAL model contribution: red supports fake, blue supports real, white near zero. The global contribution and bias are separate in scores.json. Gates are learned weights, not calibrated reliability; maps are not forgery masks or proof of fairness. Each image has its own declared display scale; use raw arrays/common scale for quantitative comparisons.\n')
    print('EXPORTED',len(rows),'diagnostic maps')


PROMPTS={
 'nochange':'Keep the supplied image exactly as it is. Preserve the same person, current apparent age, facial geometry, expression, pose, gaze, hair, facial hair, skin tone, clothing, lighting, background, framing, texture, and all existing image details. Do not beautify, retouch, smooth, sharpen, repair, or add or remove anything.',
 'younger':'Make this same person appear approximately {target_age} years old, as a younger adult. Change only age-related facial appearance. Preserve identity, expression, pose, gaze, facial geometry, hair and facial hair, skin tone, clothing, lighting, background, framing, and the original image texture. Retain natural pores. Do not beautify, repair, smooth, sharpen, add or remove objects.',
 'older':'Make this same person appear approximately {target_age} years old, as an older adult, with natural age-related changes. Change only age-related facial appearance. Preserve identity, expression, pose, gaze, facial geometry, hair and facial hair, skin tone, clothing, lighting, background, framing, and original texture. Do not beautify, repair, smooth, sharpen, add or remove objects.',
 'gender_feminine':'Edit only this same person\'s gender-related visual presentation to appear more feminine. Keep them recognizable and preserve apparent age, skin tone, bone structure, expression, gaze, pose, hairstyle, clothing, lighting, background, framing, texture and unrelated details. Use subtle changes. Do not beautify, repair, smooth, sharpen or replace the person.',
 'gender_masculine':'Edit only this same person\'s gender-related visual presentation to appear more masculine. Keep them recognizable and preserve apparent age, skin tone, bone structure, expression, gaze, pose, hairstyle, clothing, lighting, background, framing, texture and unrelated details. Use subtle changes. Do not beautify, repair, smooth, sharpen or replace the person.',
 'skin_lighter':'Make only the visible skin of this same person slightly lighter by one natural skin-tone shade. Preserve identity, apparent age, gender-related presentation, facial geometry, expression, pose, gaze, hair, clothing, lighting direction, background, framing, pores, texture and other details. Do not change ethnicity-related facial features or beautify, repair, smooth or sharpen.',
 'skin_darker':'Make only the visible skin of this same person slightly darker by one natural skin-tone shade. Preserve identity, apparent age, gender-related presentation, facial geometry, expression, pose, gaze, hair, clothing, lighting direction, background, framing, pores, texture and other details. Do not change ethnicity-related facial features or beautify, repair, smooth or sharpen.'}


def generate(args):
    # Only TRAIN is accepted here. No validation/test files or labels are read.
    rows=read_rows(args.train);attrs=args.attributes.split(',')
    if not attrs or not set(attrs).issubset({'age','gender','skin'}) or len(attrs)!=len(set(attrs)):raise ValueError('Invalid attributes')
    if args.max_originals<1:raise ValueError('Positive max-originals required')
    cells={}
    for r in rows:
        if 'age' in attrs:
            try:age=float(r.get('age_estimate',''))
            except (ValueError,TypeError):continue
            if not math.isfinite(age) or not 20<age<85:continue
        cells.setdefault((r['label'],r['gender_race']),[]).append(r)
    rng=random.Random(args.seed)
    for values in cells.values():rng.shuffle(values)
    selected=[];round_index=0;groups=sorted({key[1] for key in cells})
    while len(selected)<args.max_originals:
        found=False
        for j,group in enumerate(groups):
            label=(j+round_index)%2
            choices=cells.get((label,group)) or cells.get((1-label,group))
            if choices:selected.append(choices.pop());found=True
            if len(selected)==args.max_originals:break
        if not found:break
        round_index+=1
    if not selected:raise ValueError('No eligible source; reviewed ages/targets may be required')
    cases=[]
    for i,r in enumerate(selected):
        conditions=['nochange']
        for attr in attrs:conditions+={'age':['younger','older'],'gender':['gender_feminine','gender_masculine'],'skin':['skin_lighter','skin_darker']}[attr]
        for condition in conditions:
            target=''
            if condition in ('younger','older'):
                age=float(r['age_estimate']);target=int(max(20,age-15) if condition=='younger' else min(85,age+25))
                assert target<age if condition=='younger' else target>age
            cases.append({'pair_id':f's{i:05d}_{condition}','item':f's{i:05d}','condition':condition,'target_age':target,
                'prompt':PROMPTS[condition].format(target_age=target),'original_image':r['image'],'label':r['label'],
                'identity_ids':r['identity_ids'],'source_ids':r['source_ids'],'source_sha256':sha(r['image'])})
    plan={'version':VERSION,'model_snapshot':str(Path(args.model_snapshot).resolve()),'seed':args.seed,'train_manifest_sha256':sha(args.train),
        'steps':40,'cfg':4.,'dtype':'bfloat16','canonical_size':'source dimensions','source_count':len(selected),'cases':cases,
        'counterfactual_validity_guaranteed':False,'all_edits_independent_from_original':True,'test_read':False}
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.resume or not (out/'plan.json').exists() or json.loads((out/'plan.json').read_text())!=plan:raise ValueError('Existing outputs require identical explicit --resume')
    else:
        out=empty_out(out)
        for folder in ('native','candidates','controls'): (out/folder).mkdir()
        write(out/'plan.json',plan)
    print('PLAN',len(selected),'TRAIN originals',len(cases),'Qwen outputs, no automatic approval',flush=True)
    if args.plan_only:return
    if not Path(args.model_snapshot).is_dir():raise FileNotFoundError('Local Qwen weights required; no download')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for Qwen')
    from diffusers import QwenImageEditPlusPipeline
    done={};log=out/'generation.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            r=json.loads(line)
            if r['status']=='ok':done[r['pair_id']]=r
    pipe=QwenImageEditPlusPipeline.from_pretrained(args.model_snapshot,torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.enable_model_cpu_offload();pipe.set_progress_bar_config(disable=True)
    for i,c in enumerate(cases):
        if c['pair_id'] in done:
            row=done[c['pair_id']]
            if sha(row['counterfactual_image'])!=row['canonical_sha256'] or sha(row['native_image'])!=row['native_sha256']:raise ValueError('Completed output changed')
            continue
        native=out/'native'/(c['pair_id']+'.png');canonical=out/'candidates'/native.name
        if native.exists() or canonical.exists():raise ValueError('Unrecorded partial output preserved; explicit recovery required')
        if sha(c['original_image'])!=c['source_sha256']:raise ValueError('Source changed')
        with Image.open(c['original_image']) as im:source=im.convert('RGB');size=source.size
        start=time.perf_counter()
        with torch.inference_mode():image=pipe(image=source,prompt=c['prompt'],negative_prompt=' ',true_cfg_scale=4.,guidance_scale=1.,
            num_inference_steps=40,width=1024,height=1024,generator=torch.Generator(device='cpu').manual_seed(args.seed)).images[0].convert('RGB')
        image.save(native);image.resize(size,Image.Resampling.BOX).save(canonical)
        row={**c,'status':'ok','native_image':str(native),'counterfactual_image':str(canonical),'native_sha256':sha(native),
            'canonical_sha256':sha(canonical),'runtime_seconds':time.perf_counter()-start,'approved':0}
        with log.open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
        done[c['pair_id']]=row;print(i+1,len(cases),c['pair_id'],'ok',flush=True)
    review=[];controls=[]
    for r in selected:
        case=next(c for c in cases if c['original_image']==r['image']);path=out/'controls'/(case['item']+'_resize_vae.png')
        condition=out/'controls'/(case['item']+'_resize_condition.png')
        if not path.exists():
            with Image.open(r['image']) as im:
                matched_resize_image(im,1024**2).save(path)
        if not condition.exists():
            with Image.open(r['image']) as im:matched_resize_image(im,384**2).save(condition)
        nochange=done[next(c['pair_id'] for c in cases if c['original_image']==r['image'] and c['condition']=='nochange')]['counterfactual_image']
        controls.append({'original_image':r['image'],'original_sha256':sha(r['image']),
            'resize_control':str(path),'resize_control_sha256':sha(path),
            'condition_resize_control':str(condition),'condition_resize_control_sha256':sha(condition),
            'nochange_control':nochange,'nochange_control_sha256':sha(nochange),
            'resize_protocol':'LANCZOS input area1024^2/384^2 rounded32; BOX canonicalization; geometric only',
            'edits':[{'pair_id':c['pair_id'],'counterfactual_image':done[c['pair_id']]['counterfactual_image'],
                'sha256':done[c['pair_id']]['canonical_sha256'],'independent_from_original':True,'approved':0}
                for c in cases if c['original_image']==r['image'] and c['condition']!='nochange']})
    write(out/'resize_controls.json',controls)
    for c in cases:
        if c['condition']=='nochange':continue
        row=done[c['pair_id']];attr='age' if c['condition'] in ('younger','older') else 'gender' if c['condition'].startswith('gender') else 'skin'
        review.append({k:row[k] for k in ('pair_id','original_image','counterfactual_image','identity_ids','source_ids')}|
            {'kind':'demographic','attribute':attr,'direction':c['condition'],'approved':0,'spatial_validated':0,'approval_status':'PENDING_HUMAN_REVIEW',
            'reviewer':'','confidence':'','decision':'','unintended_changes':'','comments':''})
    review_path=out/'human_review.csv'
    if not review_path.exists():
        with review_path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(review[0]));w.writeheader();w.writerows(review)
    print('REVIEW_REQUIRED',review_path,'No generated images were trained on.',flush=True)


def training_defaults():
    return dict(seed=17,backbone='xception',weights=DEFAULT_XCEPTION,hidden=128,gate_hidden=64,size=224,
        epochs=10,batch_size=16,workers=0,mode='invariance',attributes=['age','gender','skin'],selectivity='separate',
        sampler='balanced',balance_axis='ids8',lr=.001,weight_decay=.0001,lambda_d=1.,lambda_f=1.,lambda_s=1.,lambda_map=0.,tau=.5,
        forensic_scope='fake',target_fpr=.1,min_support=20)


def smoke(device,out):
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; GPU tests not passed')
    out=empty_out(out);seed_all(17);torch.set_num_threads(4)
    model=DiagramCES().to(device);before={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith('encoder.')}
    counts=model.counts();model.train();x=torch.randn(6,3,224,224,device=device)
    output=model(x);assert output['logit'].shape==(6,) and output['patch_score'].shape[0]==6
    assert output['local_map'].shape[1:]==output['patch_gate'].shape[1:]
    torch.testing.assert_close(output['logit'],output['global_contribution']+output['local_map'].mean((1,2))+model.fusion_bias)
    assert ((output['image_gate']>0)&(output['image_gate']<1)).all() and ((output['patch_gate']>0)&(output['patch_gate']<1)).all()
    y=torch.tensor([0.,1.,1.],device=device)
    loss=ces_loss(slice_output(output,0,3),y,slice_output(output,3,5),torch.tensor([2,0]),slice_output(output,5,6),torch.tensor([2]),mode='combined')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    opt.zero_grad();loss['total'].backward()
    for branch in ('global_head','patch_head','image_gate','patch_gate'):
        grads=[p.grad for p in getattr(model,branch).parameters()];assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads)>0,branch
    assert model.fusion_raw.grad is not None and torch.isfinite(model.fusion_raw.grad).all();opt.step()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in before.items())
    for demo,forensic in [(False,False),(True,False),(False,True)]:
        o=model(x[:4]);kwargs={}
        if demo:kwargs.update(demo=slice_output(o,3,4),di=torch.tensor([0]))
        if forensic:kwargs.update(forensic=slice_output(o,3,4),fi=torch.tensor([0]))
        value=ces_loss(slice_output(o,0,3),y,mode='combined',**kwargs);opt.zero_grad();value['total'].backward()
        assert torch.isfinite(value['total']) and value['forensic_count']==0
    # Independent numerical CES reference, both real and fake signed margins, shuffled pair alignment.
    z=torch.tensor([1.,-1.,.25],device=device,requires_grad=True);zz=torch.tensor([.1,.9],device=device);zf=torch.tensor([.4,-.2],device=device)
    y2=torch.tensor([1.,0.,1.],device=device)
    l=ces_loss({'logit':z},y2,{'logit':zz},torch.tensor([2,0]),{'logit':zf},torch.tensor([0,1]),forensic_scope='all')
    a=np.array([1.,-1.,.25]);expected=np.mean(np.maximum(a,0)-a*np.array([1.,0.,1.])+np.log1p(np.exp(-abs(a))))
    expected+=np.mean(abs(np.arcsinh(a[[2,0]])-np.arcsinh([.1,.9])))
    expected+=np.maximum(0,.5-np.array([1.,-1.])*(np.arcsinh(a[[0,1]])-np.arcsinh([.4,-.2]))).mean()
    assert abs(float(l['total'].detach())-expected)<1e-6
    coupled=ces_loss({'logit':z},y2,{'logit':zz[:1]},torch.tensor([0]),{'logit':zf[:1]},torch.tensor([0]),selectivity='coupled')
    assert abs(float(coupled['coupled'].detach())-max(0,.5+abs(np.arcsinh(1)-np.arcsinh(.1))-(np.arcsinh(1)-np.arcsinh(.4))))<1e-6
    assert calibrate([.7]*4,.25)['achieved_fp']==0 and calibrate([.1,.1,.2,.9],.25)['achieved_fp']==1
    # Optional map objective: only explicitly validated spatial pairs contribute.
    om=model(x[:4]);ml=ces_loss(slice_output(om,0,3),y,slice_output(om,3,4),torch.tensor([0]),lambda_map=1.,spatial_valid=torch.tensor([False]))
    assert ml['map_count']==0 and float(ml['map_consistency'].detach())==0
    del model,opt,output,x,om,ml
    if device=='cuda':torch.cuda.empty_cache()
    fixture=out/'fixture';fixture.mkdir();fields=['image','label','identity_ids','source_ids',*AXES,'cluster','family'];pairfields=['pair_id','original_image','counterfactual_image','kind','attribute','approved','identity_ids','source_ids','forensic_validated','spatial_validated']
    rng=np.random.default_rng(17);paths={};pairs=[]
    for split in ('train','val','test'):
        rows=[]
        for i in range(4):
            image=fixture/f'{split}{i}.png';pixels=rng.integers(0,256,(48,48,3),dtype=np.uint8);Image.fromarray(pixels).save(image)
            row={'image':str(image),'label':i%2,'identity_ids':split+str(i),'source_ids':split+str(i),'cluster':split+str(i),
                'family':'synthetic',**{axis:str(i//2) for axis in AXES}};rows.append(row)
            if split=='train':
                for attr in ('age','gender','skin','forensic'):
                    if attr=='forensic' and i%2==0:continue
                    cf=fixture/f'{split}{i}_{attr}.png';Image.fromarray(np.roll(pixels,1,axis=0)).save(cf)
                    pairs.append({'pair_id':f'{i}_{attr}','original_image':str(image),'counterfactual_image':str(cf),'kind':'forensic' if attr=='forensic' else 'demographic',
                        'attribute':attr,'approved':1,'identity_ids':row['identity_ids'],'source_ids':row['source_ids'],'forensic_validated':int(attr=='forensic'),'spatial_validated':0})
        paths[split]=fixture/f'{split}.csv'
        with paths[split].open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    pp=fixture/'pairs.csv'
    with pp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=pairfields);w.writeheader();w.writerows(pairs)
    c={**training_defaults(),'train':str(paths['train']),'val':str(paths['val']),'test':'MUST_NOT_BE_OPENED','pairs':str(pp),'mode':'combined',
        'epochs':2,'batch_size':2,'min_support':1}
    ck=train_run(c,out/'tiny_training',device);first=evaluate_run(ck,paths['test'],out/'eval_before',device)
    rows=read_rows(paths['test'])
    for r in rows:r['label']=1-r['label']
    flip=fixture/'test_flipped.csv'
    with flip.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    second=evaluate_run(ck,flip,out/'eval_after',device)
    assert first['calibration']==second['calibration']
    export_maps(ck,[str(fixture/'test0.png')],out/'export_example',device)
    result={'status':'PASS','device':device,'parameters':counts,'script_sha256':sha(__file__),
        'global_patch_heads_and_gates_receive_finite_nonzero_gradients':True,'encoder_and_bn_frozen':True,
        'signed_local_map_decomposes_final_logit_with_global_and_bias':True,'mixed_empty_cuda_safe_pair_indices':True,
        'separate_and_coupled_loss_numerical_reference':True,'invalid_spatial_pairs_excluded':True,
        'tied_finite_sample_threshold_tested':True,'checkpoint_save_reload':True,'test_not_used_during_training':True,
        'threshold_unchanged_after_test_label_flip':True,'map_export_executed':True,'data':'synthetic software tests; no fairness/validity/performance conclusion'}
    write(out/'RESULT.json',result);print('DIAGRAM_SMOKE_PASS',json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--train',required=True);t.add_argument('--val',required=True);t.add_argument('--pairs');t.add_argument('--out',required=True)
    t.add_argument('--device',default='cuda');t.add_argument('--protocol-overlap-file');t.add_argument('--backbone',choices=['xception','resnet50'],default='xception')
    t.add_argument('--weights',default=DEFAULT_XCEPTION);t.add_argument('--mode',choices=['bce','invariance','sensitivity','combined'],default='invariance')
    t.add_argument('--attributes',default='age,gender,skin');t.add_argument('--selectivity',choices=['separate','coupled'],default='separate')
    t.add_argument('--sampler',choices=['balanced','natural'],default='balanced');t.add_argument('--balance-axis',choices=AXES,default='ids8')
    for name,default in [('seed',17),('epochs',10),('batch-size',16),('workers',0),('size',224),('hidden',128),('gate-hidden',64),('min-support',20)]:t.add_argument('--'+name,type=int,default=default)
    for name,default in [('lr',.001),('weight-decay',.0001),('lambda-d',1.),('lambda-f',1.),('lambda-s',1.),('lambda-map',0.),('tau',.5),('target-fpr',.1)]:t.add_argument('--'+name,type=float,default=default)
    t.add_argument('--forensic-scope',choices=['fake','all'],default='fake')
    t.add_argument('--encoder-policy',choices=['frozen','trainable'],default='frozen')
    t.add_argument('--bn-policy',choices=['frozen','update'],default='frozen')
    t.add_argument('--stage',choices=['pretrain','downstream'],default='downstream')
    t.add_argument('--encoder-lr',type=float,default=.0001);t.add_argument('--head-seed',type=int)
    t.add_argument('--encoder-checkpoint');t.add_argument('--encoder-checkpoint-sha256')
    tc=sub.add_parser('train-config');tc.add_argument('--config',required=True);tc.add_argument('--out',required=True);tc.add_argument('--device',default='cuda')
    dc=sub.add_parser('diagnose-controls');dc.add_argument('--checkpoint',required=True);dc.add_argument('--controls',required=True);dc.add_argument('--out',required=True);dc.add_argument('--device',default='cuda')
    e=sub.add_parser('evaluate');e.add_argument('--checkpoint',required=True);e.add_argument('--manifest',required=True);e.add_argument('--out',required=True);e.add_argument('--device',default='cuda')
    e=sub.add_parser('export');e.add_argument('--checkpoint',required=True);e.add_argument('--images',nargs='+',required=True);e.add_argument('--out',required=True);e.add_argument('--device',default='cuda')
    s=sub.add_parser('smoke');s.add_argument('--device',choices=['cpu','cuda'],default='cpu');s.add_argument('--out',required=True)
    g=sub.add_parser('generate');g.add_argument('--train',required=True);g.add_argument('--out',required=True);g.add_argument('--model-snapshot',default=DEFAULT_QWEN)
    g.add_argument('--max-originals',type=int,default=6);g.add_argument('--seed',type=int,default=17);g.add_argument('--attributes',default='age,gender,skin')
    g.add_argument('--plan-only',action='store_true');g.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.command=='train':
        c=vars(a).copy();c.pop('command');device=c.pop('device');out=c.pop('out');c['attributes']=[x for x in c['attributes'].split(',') if x]
        if not set(c['attributes']).issubset({'age','gender','skin'}) or len(set(c['attributes']))!=len(c['attributes']):p.error('Invalid attributes')
        train_run(c,out,device)
    elif a.command=='train-config':train_run(json.loads(Path(a.config).read_text()),a.out,a.device)
    elif a.command=='diagnose-controls':diagnose_controls(a.checkpoint,a.controls,a.out,a.device)
    elif a.command=='evaluate':evaluate_run(a.checkpoint,a.manifest,a.out,a.device)
    elif a.command=='export':export_maps(a.checkpoint,a.images,a.out,a.device)
    elif a.command=='smoke':smoke(a.device,a.out)
    else:generate(a)


if __name__=='__main__':main()
