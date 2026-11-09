"""Experimental TRAIN-source tail objective. No trainer, calibration or inference.

Source labels apply only to real TRAIN auxiliary rows. Reference tensors must be
bound to a frozen baseline and TRAIN manifest externally. The module cannot
certify source metadata or establish fairness/forensic validity.
"""
import torch
import math
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def fit_train_reference(teacher_real_logits, quantiles=(.85,.90,.95), temperature=1.):
    z=teacher_real_logits.detach().float()
    if z.ndim!=1 or len(z)<20 or not torch.isfinite(z).all():
        raise ValueError('Finite real TRAIN reference vector with adequate support required')
    if not math.isfinite(temperature) or temperature<=0 or not 0<min(quantiles)<max(quantiles)<1 or list(quantiles)!=sorted(set(quantiles)):
        raise ValueError('Invalid fixed quantiles/temperature')
    scale=torch.quantile(z,.75)-torch.quantile(z,.25)
    if not torch.isfinite(scale) or scale<=1e-6:raise ValueError('Degenerate teacher score scale')
    thresholds=torch.quantile(z,z.new_tensor(quantiles))
    targets=torch.sigmoid((z[:,None]-thresholds[None,:])/(scale*temperature)).mean(0)
    return {'thresholds':thresholds,'scale':scale,'targets':targets,'temperature':float(temperature)}


class SourceTailObjective(nn.Module):
    def __init__(self, reference, source_codes=(0,1), lambda_real=.1, lambda_fake=1., min_source_rows=16):
        super().__init__()
        if not all(math.isfinite(v) and v>=0 for v in (lambda_real,lambda_fake)) or min_source_rows<1 or len(set(source_codes))<2 or min(source_codes)<0:
            raise ValueError('Invalid coefficients/source protocol')
        for name in ('thresholds','scale','targets'):
            value=torch.as_tensor(reference[name]).detach().float().clone()
            if not torch.isfinite(value).all():raise ValueError('Nonfinite reference')
            self.register_buffer(name,value)
        if self.scale.ndim!=0 or self.scale<=0 or self.thresholds.ndim!=1 or self.thresholds.shape!=self.targets.shape or not len(self.thresholds):
            raise ValueError('Reference shape/scale mismatch')
        if ((self.targets<0)|(self.targets>1)).any():raise ValueError('Invalid reference risks')
        self.temperature=float(reference['temperature'])
        if not 0<self.temperature<float('inf'):raise ValueError('Invalid temperature')
        self.source_codes=tuple(source_codes);self.lambda_real=lambda_real;self.lambda_fake=lambda_fake
        self.min_source_rows=min_source_rows

    def forward(self, logits, labels, teacher_logits, auxiliary_real_logits, real_sources, weights=None):
        z=logits.float();y=labels.to(z.device).float();teacher=teacher_logits.detach().to(z.device).float()
        zr=auxiliary_real_logits.float();sources=real_sources.to(z.device)
        if z.ndim!=1 or z.shape!=y.shape or z.shape!=teacher.shape or not len(z) or not ((y==0)|(y==1)).all():
            raise ValueError('Classification/teacher alignment invalid')
        if zr.device!=z.device or self.scale.device!=z.device or sources.shape!=zr.shape or zr.ndim!=1 or sources.dtype!=torch.long:
            raise ValueError('Auxiliary source alignment/device invalid')
        if any(not torch.isfinite(t).all() for t in (z,teacher,zr)):raise ValueError('Nonfinite scores')
        if not set(sources.detach().cpu().tolist())<=set(self.source_codes):raise ValueError('Unknown or unverified real source')
        if any(int((sources==s).sum())<self.min_source_rows for s in self.source_codes):
            raise ValueError('Every declared real source needs the fixed auxiliary support')
        per_example=F.binary_cross_entropy_with_logits(z,y,reduction='none')
        if weights is not None:
            weights=weights.to(z.device).float()
            if weights.shape!=z.shape or not torch.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
                raise ValueError('Invalid fixed classification weights')
            per_example=per_example*weights
        bce=per_example.mean()
        exceedance=torch.sigmoid((zr[:,None]-self.thresholds[None,:])/(self.temperature*self.scale))
        rates=torch.stack([exceedance[sources==s].mean(0) for s in self.source_codes])
        source_tail=F.relu(rates-self.targets[None,:]).square().mean()
        fake=y==1
        fake_guard=F.relu((teacher[fake]-z[fake])/self.scale).square().mean() if fake.any() else z.sum()*0
        total=bce+self.lambda_real*source_tail+self.lambda_fake*fake_guard
        return {'total':total,'classification':bce,'source_tail':source_tail,'fake_guard':fake_guard,
                'source_rates':rates.detach(),'fake_count':int(fake.sum()),
                'source_counts':{s:int((sources==s).sum()) for s in self.source_codes}}
