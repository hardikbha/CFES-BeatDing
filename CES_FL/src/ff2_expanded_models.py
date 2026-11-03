"""Pinned architecture adapters for the FF2 expanded comparison.

No automatic weights/network downloads. Native scientific deviations are explicit:
F3Net uses the AI-Face FAD-only implementation at224; SPSL FFT is spatial-only;
CORE requires two views of the SAME source. UCF/PG training stays gated pending
verified specific-label/pair/SAM adapters; they are not replaced with plain BCE.
"""
import ast
import copy
import hashlib
import json
import logging
import math
import numbers
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROP = Path(__file__).resolve().parents[1] / 'proposals/ff2_expanded_baselines_20261104_r1'
TRAINABLE_METHODS = ('xception', 'efficientnetb4', 'vit_b16', 'f3net', 'spsl', 'srm', 'core', 'univfd')


def definitions(root, relative, extra=None):
    entry = json.loads((PROP / 'sources.lock.json').read_text())['repositories'][0]
    p = Path(root) / relative
    data = p.read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if blob != entry['files'][relative]['git_blob']:
        raise ValueError('Unpinned source: ' + str(p))
    namespace = dict(torch=torch, th=torch, nn=nn, F=F, np=np, math=math, numbers=numbers,
                     random=random, OrderedDict=OrderedDict, defaultdict=defaultdict,
                     Tuple=Tuple, Union=Union, AbstractDetector=nn.Module,
                     AbstractLossClass=nn.Module, Parameter=nn.Parameter,
                     logger=logging.getLogger('ff2_expanded'))
    namespace.update(extra or {})
    nodes = []
    for node in ast.parse(data).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            node = copy.deepcopy(node); node.decorator_list = []; nodes.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(p), 'exec'), namespace)
    return namespace


def xception(root, weights, channels=3, classes=1, dropout=False):
    cls = definitions(root, 'training/networks/xception.py')['Xception']
    m = cls(dict(mode='original', num_classes=classes, inc=3, dropout=dropout))
    # This adaptation exists in upstream but is unused in original mode.
    m.adjust_channel = nn.Identity()
    if weights is not None:
        sd = torch.load(weights, map_location='cpu', weights_only=True)
        sd = {k: (v[..., None, None] if 'pointwise' in k and v.ndim == 2 else v)
              for k, v in sd.items() if not k.startswith('fc.')}
        result = m.load_state_dict(sd, strict=False)
        if result.unexpected_keys or any(not k.startswith('last_linear.') for k in result.missing_keys):
            raise ValueError('Xception weights do not match the pinned ImageNet encoder')
    if channels != 3:
        w = m.conv1.weight.detach().clone()
        m.conv1 = nn.Conv2d(channels, 32, 3, 2, bias=False)
        with torch.no_grad():
            if channels == 12:
                m.conv1.weight.copy_(w.repeat(1, 4, 1, 1) / 4.)
            elif channels == 4:
                m.conv1.weight.copy_(w.mean(1, keepdim=True).repeat(1, 4, 1, 1))
            else:
                raise ValueError('Unsupported input-channel adaptation')
    return m


class BinaryLoss(nn.Module):
    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred.reshape(-1), target.float())


def spatial_phase(x):
    gray = x.mean(1, keepdim=True)
    spectrum = torch.fft.fftn(gray, dim=(-2, -1))
    return torch.fft.ifftn(torch.exp(1j * torch.angle(spectrum)), dim=(-2, -1)).real


class FrozenVisual(nn.Module):
    def __init__(self, visual):
        super().__init__(); self.visual = visual; self.visual.requires_grad_(False)
        self.fc = nn.Linear(768, 1)
        nn.init.normal_(self.fc.weight, 0., .02)

    def train(self, mode=True):
        super().train(mode); self.visual.eval(); return self

    def forward(self, x):
        with torch.no_grad():
            features = self.visual(x)
        return self.fc(features.float())


class Adapter(nn.Module):
    def __init__(self, method, root, weights=None, fixture=False):
        super().__init__(); self.method = method; self.fixture = fixture
        if method not in TRAINABLE_METHODS:
            raise ValueError('UCF/PG require the declared specific-label and paired-training adapter; plain BCE substitution forbidden')
        if not fixture and (not weights or not Path(weights).is_file()):
            raise FileNotFoundError('Supply verified local ImageNet/CLIP weights; no automatic download')
        if method == 'xception':
            self.net = xception(root, weights)
        elif method == 'efficientnetb4':
            from efficientnet_pytorch import EfficientNet
            self.net = EfficientNet.from_name('efficientnet-b4')
            if weights:
                self.net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
            self.net._fc = nn.Linear(1792, 1)
            # Preserve the pretrained stem. AI-Face replaces it randomly; this
            # deliberate correction is named in the method configuration.
        elif method == 'vit_b16':
            from torchvision.models import vit_b_16
            self.net = vit_b_16(weights=None)
            if weights:
                self.net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
            self.net.heads[0] = nn.Linear(768, 1)
        elif method == 'univfd':
            defs = definitions(root, 'training/models/clip/model.py')
            visual = defs['VisionTransformer'](input_resolution=224, patch_size=14, width=1024, layers=24, heads=16, output_dim=768)
            if weights:
                # Accept the official local JIT checkpoint or a tensor-only
                # state dict. Discard the unused text tower for deployment.
                try:
                    state = torch.jit.load(str(weights), map_location='cpu').state_dict()
                except RuntimeError:
                    state = torch.load(weights, map_location='cpu', weights_only=True)
                state = {k[len('visual.'):]: v for k, v in state.items() if k.startswith('visual.')}
                visual.load_state_dict(state, strict=True)
            self.net = FrozenVisual(visual)
        else:
            native_names = {'f3net': ('f3net_detector.py', 'F3netDetector'),
                            'spsl': ('spsl_detector.py', 'SpslDetector'),
                            'srm': ('srm_detector.py', 'SRMDetector'),
                            'core': ('core_detector.py', 'CoreDetector')}
            file, name = native_names[method]
            losses = {'bce': BinaryLoss}
            if method == 'srm':
                losses['am_softmax'] = definitions(root, 'training/loss/am_softmax.py')['AMSoftmaxLoss']
            if method == 'core':
                losses['consistency_loss'] = definitions(root, 'training/loss/consistency_loss.py')['ConsistencyCos']
            defs = definitions(root, 'training/detectors/' + file, {'LOSSFUNC': losses})
            if method == 'srm':
                # Original learned spatial attention assumes256px/16x16.
                # The common224px recipe produces14x14 at both CMA stages.
                defs['DualCrossModalAttention'].__init__.__defaults__ = (None, 14, 8, False)
            cls = defs[name]
            channels = 12 if method == 'f3net' else 4 if method == 'spsl' else 3
            nclass = 2 if method in ('srm', 'core') else 1
            cls.build_backbone = lambda _: xception(root, weights, channels, nclass, .5 if method == 'f3net' else False)
            self.net = cls()
            if method == 'f3net':
                self.net.FAD_head = defs['FAD_Head'](224)
            if method == 'spsl':
                self.net.phase_without_amplitude = spatial_phase
            if method == 'srm':
                # The second classifier is unused by the two-stream fusion.
                self.net.backbone_srm.last_linear = nn.Identity()

    def predict_dict(self, x):
        if self.method == 'xception':
            return {'cls': self.net(x)[0]}
        if self.method in ('efficientnetb4', 'vit_b16', 'univfd'):
            return {'cls': self.net(x)}
        return self.net({'image': x}, inference=True)

    def forward(self, x):
        pred = self.predict_dict(x)['cls']
        return pred[:, 1] - pred[:, 0] if pred.shape[1] == 2 else pred[:, 0]

    def loss(self, x, y, second_view=None):
        if self.method == 'core':
            if second_view is None or second_view.shape != x.shape:
                raise ValueError('CORE requires two aligned views of the same source images')
            pred = self.predict_dict(torch.cat((x, second_view)))
            return self.net.get_losses({'label': torch.cat((y, y))}, pred)['overall']
        if self.method in ('f3net', 'spsl', 'srm'):
            return self.net.get_losses({'label': y}, self.predict_dict(x))['overall']
        return F.binary_cross_entropy_with_logits(self(x), y.float())

    def counts(self):
        return {'deployed_total_parameters': sum(p.numel() for p in self.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'frozen_parameters': sum(p.numel() for p in self.parameters() if not p.requires_grad)}
