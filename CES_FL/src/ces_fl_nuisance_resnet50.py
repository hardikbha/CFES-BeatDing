#!/usr/bin/env python3
"""Bounded ResNet L_H nuisance-augmentation pilot, 4 November 2026.

Derived from ces_fl_head_ablation_resnet50.py at 08f78b3, SHA256
72d368a439658c601a5fe4d717237b1561190e85c13c70f6db3a441bc5cdbe78.
New config nuisance_mode=none/matched changes TRAIN image processing only.
Matched mild transformations apply to originals of BOTH classes and to the
same source's sham/treated views. Validation/calibration remain unaugmented.
The four-arm pilot isolates augmentation and H's whole objective package.
No explicit feature disentanglement or real-forgery validity is established.

New standalone variant of forensic-response v3 (source SHA256
2549cb1345d1e2ee617e8a77ca288a0acbbfd05cf480f7b91a46b8627fcb74b2).
The downstream model has exactly one Linear(2048,1) prediction head; the
nonlinear heads, gates and fusion used during detection pretraining are not
imported into the downstream comparison.
The encoder, losses, source exposure, selection and calibration are unchanged.
Linear maps are w^T F at each location; their mean plus classifier bias is z.
No learned reliability, causal localization, fairness or novelty is claimed.

Dependencies: Python>=3.10, torch, numpy, Pillow, scikit-learn. No project
imports, downloads, diffusion models, automatic cluster jobs or test inference.
Commands: bind, preflight, preview, qualify, smoke, dry-run, train, evaluate,
compare, export. Run --help and each subcommand --help for the explicit stages.
Synthetic local compositing interventions are SBI-inspired engineering probes,
not an official SBI reproduction or validated demographic counterfactuals.
All originals get BCE. Synthetic views get response losses only, never assumed
benchmark authenticity labels. Group fairness is measured, never guaranteed.
Historical protocols/results remain separate. Encoder checkpoint is required.

This file preserves the original Xception implementation below only so the
scientific loss/data path stays reviewable against ces_fl_head_ablation.py.
Detector construction is deliberately ResNet-50 only. It uses torchvision's
spatial 2048-channel trunk, ImageNet normalization, layer4 fine-tuning, a
single Linear(2048,1) classifier, and frozen BatchNorm running statistics.
"""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from contextlib import nullcontext

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

VERSION = 'CESFL_NUISANCE_RESNET50_v1'
AXES = ('gender', 'race', 'age_bin', 'skin_bin', 'gender_race', 'ids8')
ARMS = ('A', 'B', 'C', 'D')
OPERATORS = ('local_warp', 'local_resample')
STRENGTHS = (0.5, 1.0)

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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, p)


def atomic_save(path, obj):
    p = Path(path); tmp = p.with_suffix('.tmp')
    torch.save(obj, tmp); os.replace(tmp, p)


def fresh_dir(path):
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise ValueError(f'Preserve existing output: {p}')
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_for(*values):
    return int(digest(values)[:15], 16)


def ids(value):
    return {v.strip() for v in str(value).split(';') if v.strip()}


def read_manifest(path, allow_duplicates=False):
    p = Path(path)
    with p.open(newline='') as f:
        reader = csv.DictReader(f)
        required = {'image', 'label', 'identity_ids', 'source_ids'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing manifest columns: {required}')
        rows = list(reader)
    seen = {}
    for r in rows:
        image = Path(r['image'])
        image = (p.parent / image).resolve() if not image.is_absolute() else image.resolve()
        if r['label'] not in ('0', '1') or not image.is_file():
            raise ValueError(f'Invalid image/label: {image}')
        r['image'], r['label'] = str(image), int(r['label'])
        for field in ('identity_ids', 'source_ids'):
            if not ids(r[field]): raise ValueError(f'Empty {field}')
            r[field] = ';'.join(sorted(ids(r[field])))
        for a in AXES:
            r[a] = (r.get(a) or 'unknown').strip() or 'unknown'
        r['family'] = r.get('family') or 'unknown'
        signature = tuple(r.get(k) for k in ('label', 'identity_ids', 'source_ids', *AXES, 'family'))
        if r['image'] in seen and (not allow_duplicates or seen[r['image']] != signature):
            raise ValueError('Duplicate original or conflicting duplicate metadata')
        seen[r['image']] = signature
    if {r['label'] for r in rows} != {0, 1}: raise ValueError('Both original classes required')
    return rows


def split_audit(train, val, overlap_file=None, evaluation=False):
    if {r['image'] for r in train} & {r['image'] for r in val}:
        raise ValueError('Original image path leakage')
    declared = json.loads(Path(overlap_file).read_text()).get('declared_overlap_values', {}) if overlap_file else {}
    if not isinstance(declared, dict): raise ValueError('Overlap declarations must be a mapping')
    result = {}
    suffix = 'fit:test' if evaluation else 'train:val'
    expected_keys = {f'{field}:{suffix}' for field in ('identity_ids', 'source_ids')}
    # Historical TRAIN/VAL audits also inventory TRAIN/TEST and VAL/TEST.
    # Recognize those keys without checking or using their values in training.
    # TEST evaluation still needs an explicit fit:TEST (TRAIN+VAL) inventory.
    allowed_keys = expected_keys if evaluation else {
        f'{field}:{pair}' for field in ('identity_ids', 'source_ids')
        for pair in ('train:val', 'train:test', 'val:test')}
    if set(declared) - allowed_keys: raise ValueError('Unexpected overlap exception keys')
    for field in ('identity_ids', 'source_ids'):
        overlap = sorted(set().union(*(ids(r[field]) for r in train)) & set().union(*(ids(r[field]) for r in val)))
        if overlap != declared.get(f'{field}:{suffix}', []):
            raise ValueError(f'Unrecorded or changed {field} overlap')
        result[field] = len(overlap)
    return {'recorded_contributor_overlap': result, 'true_person_separation_verified': False,
            'checked_overlap_keys': sorted(expected_keys),
            'deferred_overlap_keys': sorted(set(declared) - expected_keys)}


def calibrate(real, target=0.1):
    z = np.asarray(real, np.float64)
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all() or not 0 <= target < 1:
        raise ValueError('Invalid calibration input')
    f = Fraction(str(target)) * len(z); budget = f.numerator // f.denominator
    threshold = float(np.nextafter(np.sort(z)[::-1][budget], np.inf))
    if not math.isfinite(threshold): raise ValueError('Nonfinite threshold')
    fp = int((z >= threshold).sum())
    return {'threshold_logit': threshold, 'n_real': len(z), 'allowed_fp': budget,
            'achieved_fp': fp, 'achieved_fpr': fp / len(z), 'requested_fpr': target,
            'test_fpr_guaranteed': False}


class Detector(nn.Module):
    def __init__(self, hidden=128, gate_hidden=64, policy='exit', head_type='linear'):
        super().__init__()
        if policy not in ('frozen', 'exit', 'full'): raise ValueError('Unknown encoder policy')
        if head_type != 'linear': raise ValueError('ResNet Table 3 comparison requires the plain linear head')
        self.head_type = head_type
        from torchvision.models import resnet50
        base = resnet50(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.classifier = nn.Linear(2048, 1)
        self.policy = policy
        for name, p in self.encoder.named_parameters():
            # torchvision Sequential child 7 is layer4.
            p.requires_grad_(policy == 'full' or (policy == 'exit' and name.startswith('7.')))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # All downstream arms use frozen BN running buffers, including full tuning.
        self.encoder.eval()
        return self

    def forward(self, x):
        with torch.no_grad() if self.policy == 'frozen' else nullcontext():
            f = self.encoder(x)
        z = self.classifier(f.mean((2, 3))).squeeze(1)
        spatial = (f * self.classifier.weight[0][None, :, None, None]).sum(1)
        return {'logit': z, 'local_map': spatial,
                'global_contribution': torch.zeros_like(z), 'bias': self.classifier.bias[0]}

    def counts(self):
        return {'total': sum(p.numel() for p in self.parameters()),
                'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)}

    def parameter_report(self):
        encoder = sum(p.numel() for p in self.encoder.parameters())
        return {**self.counts(), 'encoder': encoder, 'head': self.counts()['total'] - encoder,
                'encoder_trainable': sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
                'head_type': self.head_type, 'encoder_policy': self.policy,
                'backbone': 'resnet50', 'trainable_encoder_scope': 'layer4'}


def load_initial(c):
    seed_all(c['seed'])
    model = Detector(c['hidden'], c['gate_hidden'], c['encoder_policy'], c['head_type'])
    ck = torch.load(c['init_checkpoint'], map_location='cpu', weights_only=True)
    if bool(ck.get('synthetic_fixture', False)) != c['synthetic_fixture']:
        raise ValueError('Synthetic checkpoint cannot initialize a production run')
    if ck.get('format') not in ('CESFL_RESNET50_P50_v1', 'CESFL_DIAGRAM_v0.2.1',
                                'CESFL_DIAGRAM_v0.2', 'CESFL_CF5050_v1'):
        raise ValueError('Only explicit CES-FL detection-pretraining formats supported')
    pc = ck['config']
    if ck.get('stage', pc.get('stage')) != 'pretrain' or pc.get('backbone') != 'resnet50':
        raise ValueError('Initialization must be ResNet-50 detection pretraining')
    # Check original source manifests in addition to the immutable checkpoint hash.
    for name in ('train', 'val'):
        if Path(pc[name]).resolve() != Path(c[name]).resolve():
            raise ValueError('Pretraining dataset paths differ; require a separately audited export')
        if sha(pc[name]) != c[name + '_sha256']: raise ValueError('Pretraining input mismatch')
        saved = pc.get('input_hashes', {}).get(name)
        if saved and saved != c[name + '_sha256']: raise ValueError('Stored pretraining hash mismatch')
    state = {k[len('encoder.'):]: v for k, v in ck['model'].items() if k.startswith('encoder.')}
    model.encoder.load_state_dict(state, strict=True)
    # Fresh heads/gates are identical across A/B/C/D for the same downstream seed.
    return model


def load_pixels(row, size):
    data = Path(row['image']).read_bytes()
    if row.get('image_sha256') and hashlib.sha256(data).hexdigest() != row['image_sha256']:
        raise ValueError('Image bytes changed since preflight')
    with Image.open(io.BytesIO(data)) as im:
        a = np.asarray(im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR), np.float32).copy() / 255.
    return torch.from_numpy(a).permute(2, 0, 1)


def normalize(x):
    mean = x.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = x.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return (x - mean) / std


NUISANCE_BANK = ('identity', 'jpeg90', 'resize90', 'blur035')


def nuisance_view(x, case):
    """A deterministic CPU transform of RGB pixels in [0,1]; no class/group input.

    These are nuisance candidates, not validated label-preserving transformations.
    Apply the SAME case to x/sham/treated AFTER the procedural intervention.
    """
    if case not in NUISANCE_BANK or x.device.type != 'cpu' or x.ndim != 3 or x.shape[0] != 3:
        raise ValueError('Invalid nuisance input/case')
    if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError('Nuisance input must be finite RGB pixels in [0,1]')
    if case == 'identity':
        return x.clone()
    if case == 'resize90':
        h, w = x.shape[-2:]
        small = F.interpolate(x[None], size=(max(2, round(.9*h)), max(2, round(.9*w))),
                              mode='bilinear', align_corners=False)
        return F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)[0].clamp(0, 1)
    image = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    if case == 'jpeg90':
        buf = io.BytesIO(); image.save(buf, 'JPEG', quality=90, subsampling=0)
        buf.seek(0)
        with Image.open(buf) as decoded:
            pixels = np.asarray(decoded.convert('RGB'), np.float32).copy() / 255.
    else:
        pixels = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=.35)), np.float32).copy() / 255.
    return torch.from_numpy(pixels).permute(2, 0, 1)


def nuisance_case(c, epoch, *key):
    if c['nuisance_mode'] == 'none':
        return 'identity'
    # Independent of labels, groups, loss values, and model predictions.
    return NUISANCE_BANK[(seed_for(c['seed'], 'nuisance-v1', *key) + epoch) % len(NUISANCE_BANK)]


def common_nuisance(x, sham, treated, case):
    return tuple(nuisance_view(v, case) for v in (x, sham, treated))


def intervention(x, operator, strength, seed):
    """Procedural probes, NOT official SBI or validated real-world forgery synthesis.

    Same shared common JPEG transform, mask, blend and resampling implementation.
    Sham uses zero local mismatch. Local changes may differ in perceptual difficulty
    by group even at identical nominal strength. No global JPEG/blur fake labels.
    """
    if operator not in OPERATORS or strength < 0 or strength > 1: raise ValueError('Invalid intervention')
    if x.device.type != 'cpu': raise ValueError('Procedural operator runs on CPU')
    rng = random.Random(seed); h, w = x.shape[-2:]
    quality = rng.choice((85, 95, 100))
    buf = io.BytesIO()
    Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    with Image.open(buf) as im:
        common = torch.from_numpy(np.asarray(im.convert('RGB'), np.float32).copy() / 255.).permute(2, 0, 1)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
    cx, cy = rng.uniform(-.10, .10), rng.uniform(-.12, .05)
    rx, ry = rng.uniform(.40, .55), rng.uniform(.50, .65)
    distance = ((xx - cx) / rx).square() + ((yy - cy) / ry).square()
    mask = ((1.0 - distance) / .25).clamp(0, 1)[None]
    base_grid = torch.stack((xx, yy), -1)[None]
    # Same identity-resampling path for the sham in both operator families.
    identity = F.grid_sample(common[None], base_grid, mode='bilinear', padding_mode='border', align_corners=True)[0]
    if operator == 'local_warp':
        dx, dy = rng.choice((-1, 1)) * .045 * strength, rng.choice((-1, 1)) * .035 * strength
        grid = base_grid + torch.tensor([dx, dy])
        source = F.grid_sample(common[None], grid, mode='bilinear', padding_mode='border', align_corners=True)[0]
    else:
        ratio = 1.0 - .55 * strength
        small = F.interpolate(identity[None], size=(max(2, round(h * ratio)), max(2, round(w * ratio))), mode='bilinear', align_corners=False)
        source = F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)[0]
    if strength == 0: source = identity.clone()
    sham = (1 - mask) * common + mask * identity
    treated = (1 - mask) * common + mask * source
    return sham.clamp(0, 1), treated.clamp(0, 1), mask, {'operator': operator, 'strength': strength, 'seed': seed, 'jpeg_quality': quality}


class Originals(Dataset):
    def __init__(self, rows, c, epoch=0, pairs=False, training=False):
        self.rows, self.c, self.epoch, self.pairs, self.training = rows, c, epoch, pairs, training
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]; x = load_pixels(row, self.c['size'])
        s, t = x, x
        if self.pairs and row['label'] == 0:
            case = (seed_for(self.c['seed'], row['image']) + self.epoch) % (len(OPERATORS) * len(STRENGTHS))
            op, strength = OPERATORS[case // len(STRENGTHS)], STRENGTHS[case % len(STRENGTHS)]
            s, t, _, _ = intervention(x, op, strength, seed_for(self.c['seed'], row['image'], self.epoch, 'pair'))
        case = nuisance_case(self.c, self.epoch, 'main', row['image']) if self.training else 'identity'
        x, s, t = common_nuisance(x, s, t, case)
        return {'x': normalize(x), 's': normalize(s), 't': normalize(t), 'y': row['label'], 'index': i,
                'nuisance_index': NUISANCE_BANK.index(case)}


def group_inventory(rows, c):
    active = {}; report = {}
    for axis in c['axes']:
        groups = defaultdict(lambda: {0: [], 1: []})
        for i, row in enumerate(rows): groups[row[axis]][row['label']].append(i)
        active[axis] = {g: cell for g, cell in sorted(groups.items()) if g.lower() != 'unknown' and min(len(cell[0]), len(cell[1])) >= c['min_train_support']}
        report[axis] = {g: {'real': len(cell[0]), 'fake': len(cell[1]), 'eligible': g in active[axis]} for g, cell in sorted(groups.items())}
        if len(active[axis]) < 2: raise ValueError(f'Insufficient supported groups for requested axis {axis}')
    if c['error_control'] == 'hierarchical_real':
        # Validate metadata once, using TRAIN only. No visual label inference.
        known = [r for r in rows if all(r[k].lower() != 'unknown' for k in ('gender', 'race', 'gender_race'))]
        if any(r['gender_race'] != r['gender'] + '-' + r['race'] for r in known):
            raise ValueError('TRAIN joint/parent demographic metadata disagree')
        if {r['gender_race'] for r in known} != set(active['gender_race']):
            raise ValueError('Missing/sparse TRAIN joint cell; no coverage reduction')
        hierarchy_layout(rows, active['gender_race'])
    return active, report


def auxiliary(rows, active, c, event, epoch):
    axis = c['axes'][event % len(c['axes'])]
    groups = active[axis]; n = c['group_per_class']
    case = (event // len(c['axes']) + epoch) % (len(OPERATORS) * len(STRENGTHS))
    op, strength = OPERATORS[case // len(STRENGTHS)], STRENGTHS[case % len(STRENGTHS)]
    rng = random.Random(seed_for(c['seed'], epoch, event, 'aux'))
    indices, labels, group_codes, originals, sham, treated = [], [], [], [], [], []
    support_fraction = []
    group_names = list(groups)
    for gi, name in enumerate(group_names):
        cell_support = []
        for label in (0, 1):
            pool = groups[name][label]
            chosen = rng.sample(pool, n) if len(pool) >= n else rng.choices(pool, k=n)
            # TRAIN records only. This measures recorded contributor-tuple
            # diversity, not independent biological identities or components.
            tokens = {recorded_contributor(rows[ix]) for ix in chosen}
            cell_support.append(len(tokens) / len(chosen))
            for ix in chosen:
                x = load_pixels(rows[ix], c['size'])
                # One common augmentation case across every group/class in this event.
                case_nuisance = nuisance_case(c, epoch, 'aux', event)
                originals.append(normalize(nuisance_view(x, case_nuisance)))
                indices.append(ix); labels.append(label); group_codes.append(gi)
                if label == 0:
                    # Same nominal operator, strength and nuisance/geometry draw across groups.
                    s, t, _, _ = intervention(x, op, strength, seed_for(c['seed'], epoch, event, 'matched-nuisance'))
                    sham.append(normalize(nuisance_view(s, case_nuisance)))
                    treated.append(normalize(nuisance_view(t, case_nuisance)))
        support_fraction.append(cell_support)
    counts = torch.tensor([[len(groups[g][label]) for label in (0, 1)] for g in group_names], dtype=torch.float32)
    priors = counts / counts.sum(0)
    result = {'x': torch.stack(originals), 's': torch.stack(sham), 't': torch.stack(treated),
            'y': torch.tensor(labels), 'group': torch.tensor(group_codes), 'priors': priors,
            'support_fraction': torch.tensor(support_fraction, dtype=torch.float32),
            'axis': axis, 'groups': group_names, 'indices': indices, 'case': case,
            'operator': op, 'strength': strength, 'nuisance_case': case_nuisance}
    if c['error_control'] == 'hierarchical_real' and axis == 'gender_race':
        result['hierarchy'] = hierarchy_layout(rows, groups)
    return result


def hierarchy_layout(rows, groups):
    """Class0 TRAIN within-parent race proportions, in actual auxiliary order.

    group_inventory checks every TRAIN row first. Here representatives recover
    that verified metadata without rescanning all TRAIN frames each auxiliary step.
    Sparse/missing Cartesian cells fail; unknown rows still retain original BCE.
    """
    names = list(groups)
    if not names or any(not groups[g][0] or not groups[g][1] for g in names):
        raise ValueError('Hierarchy needs both classes in every joint cell')
    pairs = [(rows[groups[g][0][0]]['gender'], rows[groups[g][0][0]]['race']) for g in names]
    parents, races = sorted({g for g, _ in pairs}), sorted({r for _, r in pairs})
    if len(parents) < 2 or len(races) < 2 or len(set(pairs)) != len(pairs):
        raise ValueError('Missing/duplicate hierarchy parent or race cells')
    if set(pairs) != {(g, r) for g in parents for r in races}:
        raise ValueError('Incomplete TRAIN joint Cartesian coverage')
    codes = torch.tensor([parents.index(g) for g, _ in pairs], dtype=torch.long)
    nreal = torch.tensor([len(groups[g][0]) for g in names], dtype=torch.float32)
    within = torch.stack([torch.where(codes == h, nreal, 0.) for h in range(len(parents))])
    within = within / within.sum(1, keepdim=True)
    return {'parents': parents, 'parent_codes': codes, 'within_parent_real_priors': within}


def hierarchical_real_error(z, y, group, priors, layout, threshold, temperature):
    """Replace ONLY the gender_race auxiliary real summand of D's error loss.

    r_j = E[sigmoid((z-t_TRAIN)/T)|real,j]. Parent risks use TRAIN real
    within-gender race proportions. b=stopgrad(sum_j pi_j,real*r_j).
    H = .5*(2 mean_j relu(r_j-b)) + .5*(2 mean_s relu(R_s-b)).
    Return .5*H + .5*mean_j abs(FNRsoft_j - attached pooled FNRsoft).
    Fake-image value/gradient is exactly the original D term. All other axes
    retain D's full legacy term; no diversity discount or alignment ramp.
    """
    n = len(priors)
    if priors.shape != (n, 2) or not torch.isfinite(priors).all() or (priors < 0).any():
        raise ValueError('Invalid joint class priors')
    if not torch.allclose(priors.sum(0), torch.ones(2, device=priors.device)):
        raise ValueError('Joint class priors must sum to one')
    codes = layout['parent_codes'].to(device=z.device)
    q = layout['within_parent_real_priors'].to(device=z.device, dtype=torch.float32)
    if codes.shape != (n,) or codes.dtype != torch.long or q.ndim != 2 or q.shape[1] != n or q.shape[0] < 2:
        raise ValueError('Hierarchy alignment/shape mismatch')
    if not torch.isfinite(q).all() or (q < 0).any() or (codes < 0).any() or (codes >= len(q)).any():
        raise ValueError('Invalid hierarchy weights/codes')
    if not torch.allclose(q.sum(1), torch.ones(len(q), device=z.device)):
        raise ValueError('Within-parent weights must sum to one')
    expected = torch.stack([torch.where(codes == h, priors[:, 0].to(z.device), 0.) for h in range(len(q))])
    if (expected.sum(1) <= 0).any():
        raise ValueError('Missing parent support')
    expected = expected / expected.sum(1, keepdim=True)
    if not torch.allclose(q, expected):
        raise ValueError('Parent weights differ from TRAIN class0 proportions')
    if z.ndim != 1 or y.shape != z.shape or group.shape != z.shape or not torch.isfinite(z).all():
        raise ValueError('Invalid joint logits/labels/alignment')
    if not ((y == 0) | (y == 1)).all() or (group < 0).any() or (group >= n).any():
        raise ValueError('Invalid joint label/group code')
    p = torch.sigmoid((z.float() - float(threshold)) / temperature)
    real, fake = [], []
    for j in range(n):
        a, b = (group == j) & (y == 0), (group == j) & (y == 1)
        if not a.any() or not b.any():
            raise ValueError('Missing auxiliary joint/class cell')
        real.append(p[a].mean()); fake.append((1-p[b]).mean())
    real, fake = torch.stack(real), torch.stack(fake)
    priors = priors.to(z.device)
    reference = (real * priors[:, 0]).sum().detach()
    # Elementwise float32 reduction avoids autocast downcasting a matrix product.
    parent = (q * real[None, :]).sum(1)
    hierarchy = F.relu(real-reference).mean() + F.relu(parent-reference).mean()
    fake_reference = (fake * priors[:, 1]).sum()  # Attached, exactly as D.
    return .5 * hierarchy + .5 * (fake-fake_reference).abs().mean()


def recorded_contributor(row):
    identity = tuple(sorted(ids(row.get('identity_ids', ''))))
    source = tuple(sorted(ids(row.get('source_ids', ''))))
    return (identity, source) if identity or source else ('image-fallback', row['image'])


def support_upper_error(z, y, group, priors, support_fraction, threshold, temperature):
    """One-sided soft error control; better groups are never pushed toward error.

    The reference is the detached TRAIN-prior-weighted auxiliary estimate.
    U/n discounts repeated recorded contributors in each TRAIN-only cell.
    Factor2 matches the legacy value for uniform priors/full support, since
    sum positive deviations is half the sum absolute deviations in that case.
    Small balanced batches and soft/hard tail mismatch remain limitations.
    """
    if support_fraction.shape != priors.shape:
        raise ValueError('Support/prior alignment mismatch')
    rho = support_fraction.to(device=z.device, dtype=torch.float32)
    if not torch.isfinite(rho).all() or (rho <= 0).any() or (rho > 1).any():
        raise ValueError('Invalid TRAIN contributor support fraction')
    p = torch.sigmoid((z.float() - float(threshold)) / temperature)
    means = []
    for g in range(len(priors)):
        real, fake = (group == g) & (y == 0), (group == g) & (y == 1)
        if not real.any() or not fake.any():
            raise ValueError('Missing auxiliary class/group support')
        means.append(torch.stack((p[real].mean(), (1-p[fake]).mean())))
    errors = torch.stack(means)
    reference = (errors * priors.to(z.device)).sum(0).detach()
    return 2 * (rho * F.relu(errors-reference)).mean()


def alignment_multiplier(epoch, warmup_epochs):
    if type(epoch) is not int or epoch < 0 or type(warmup_epochs) is not int or warmup_epochs < 0:
        raise ValueError('Invalid alignment schedule')
    return 1. if warmup_epochs == 0 else min(1., (epoch+1)/warmup_epochs)


def soft_error_disparity(z, y, group, priors, threshold, temperature):
    """Normalized soft FPR+FNR deviation from TRAIN-prior-weighted group means.
    Threshold is detached and TRAIN-derived. Unknown/unsupported cells excluded.
    This is a surrogate, not Ding's thresholded F_FPR or a guarantee of parity.
    """
    p = torch.sigmoid((z.float() - float(threshold)) / temperature)
    means = []
    for g in range(len(priors)):
        real, fake = (group == g) & (y == 0), (group == g) & (y == 1)
        if not real.any() or not fake.any(): raise ValueError('Missing auxiliary class/group support')
        means.append(torch.stack((p[real].mean(), (1 - p[fake]).mean())))
    m = torch.stack(means); pooled = (m * priors.to(z.device)).sum(0)
    return (m - pooled).abs().mean()


def response(sham_logits, treated_logits):
    return torch.asinh(treated_logits.float() - sham_logits.float())


def sensitivity(r, margin): return F.relu(margin - r).mean() if r.numel() else r.sum() * 0


def response_alignment(r, groups):
    """Equal-count 1D quantile alignment within ONE operator/strength stratum.
    Each group contributes equally. This is no same-person demographic edit.
    """
    codes = torch.unique(groups, sorted=True)
    if len(codes) < 2: raise ValueError('Response alignment needs >=2 groups')
    samples = [torch.sort(r[groups == g]).values for g in codes]
    if len({len(s) for s in samples}) != 1 or not samples[0].numel():
        raise ValueError('Quantile alignment requires matched nonempty group counts')
    q = torch.stack(samples)
    return (q - q.mean(0, keepdim=True)).abs().mean()


def autocast(device, c):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and c['amp'] else nullcontext()


def train_step(model, b, aux, weights, c, arm, threshold, device, epoch=0):
    x, y, ix = b['x'].to(device), b['y'].to(device).float(), b['index']
    with autocast(device, c):
        z = model(x)['logit']
        cls = (F.binary_cross_entropy_with_logits(z.float(), y, reduction='none') * weights[ix].to(device)).mean()
        zero = z.float().sum() * 0
        fair = zero; orig_sens = zero; aux_sens = zero; align = zero; paired_count = 0
        orig_control = zero; aux_control = zero
        if arm in ('C', 'D'):
            real = y == 0
            if real.any():
                zs = model(b['s'][real.cpu()].to(device))['logit']
                zt = model(b['t'][real.cpu()].to(device))['logit']
                orig_sens = sensitivity(response(zs, zt), c['margin']); paired_count = int(real.sum())
                orig_control = (torch.asinh(zs.float()) - torch.asinh(z[real].float())).abs().mean()
        if aux is not None and arm != 'A':
            za = model(aux['x'].to(device))['logit']
            ay, ag = aux['y'].to(device), aux['group'].to(device)
            if c['error_control'] == 'hierarchical_real' and aux['axis'] == 'gender_race':
                fair = hierarchical_real_error(za, ay, ag, aux['priors'], aux['hierarchy'], threshold, c['temperature'])
            elif c['error_control'] == 'support_upper':
                fair = support_upper_error(za, ay, ag, aux['priors'], aux['support_fraction'], threshold, c['temperature'])
            else:
                fair = soft_error_disparity(za, ay, ag, aux['priors'], threshold, c['temperature'])
            if arm in ('C', 'D'):
                zs = model(aux['s'].to(device))['logit']; zt = model(aux['t'].to(device))['logit']
                rr = response(zs, zt); aux_sens = sensitivity(rr, c['margin'])
                aux_control = (torch.asinh(zs.float()) - torch.asinh(za[ay == 0].float())).abs().mean()
                if arm == 'D': align = response_alignment(rr, ag[ay == 0])
        # aux_every compensates deterministic periodic evaluation frequency.
        total = cls + c['lambda_sensitivity'] * (orig_sens + c['sham_weight'] * orig_control)
        if aux is not None:
            total = total + c['aux_every'] * (c['lambda_fair'] * fair + c['lambda_sensitivity'] * (aux_sens + c['sham_weight'] * aux_control) + c['lambda_alignment'] * alignment_multiplier(epoch, c['alignment_warmup_epochs']) * align)
    return {'total': total, 'classification': cls, 'error_fairness': fair,
            'original_sensitivity': orig_sens, 'aux_sensitivity': aux_sens,
            'response_alignment': align, 'original_control_stability': orig_control,
            'aux_control_stability': aux_control}, paired_count


def original_weights(rows, axis):
    cells = Counter((r['label'], r[axis]) for r in rows)
    # Every original once/epoch. Fixed inverse-cell weights, no per-batch renormalization.
    return torch.tensor([len(rows) / (len(cells) * cells[r['label'], r[axis]]) for r in rows], dtype=torch.float32)


def make_loader(rows, c, epoch=0, pairs=False, shuffle=False):
    return DataLoader(Originals(rows, c, epoch, pairs, training=shuffle), batch_size=c['batch_size'], shuffle=shuffle,
                      num_workers=c['workers'], pin_memory=False, drop_last=False,
                      generator=torch.Generator().manual_seed(c['seed'] + epoch))


@torch.inference_mode()
def predict(model, rows, c, device):
    model.eval(); values = []
    for b in make_loader(rows, c):
        # Float32 evaluation/calibration intentionally independent of training AMP.
        values.extend(model(b['x'].to(device))['logit'].float().cpu().tolist())
    z = np.asarray(values, np.float64)
    if not np.isfinite(z).all(): raise RuntimeError('Nonfinite predictions')
    return z


def metric_report(rows, logits, threshold, min_support, eligible=None):
    from sklearn.metrics import roc_auc_score
    z = np.asarray(logits, np.float64); y = np.asarray([r['label'] for r in rows]); pred = z >= threshold
    if z.shape != y.shape or not np.isfinite(z).all(): raise ValueError('Invalid score alignment')
    def one(mask):
        yy, zz, pp = y[mask], z[mask], pred[mask]; real, fake = yy == 0, yy == 1
        return {'n': int(len(yy)), 'n_real': int(real.sum()), 'n_fake': int(fake.sum()),
                'auc': float(roc_auc_score(yy, zz)) if real.any() and fake.any() else None,
                'fpr': float(pp[real].mean()) if real.any() else None,
                'fnr': float((~pp[fake]).mean()) if fake.any() else None,
                'positive_rate': float(pp.mean()) if len(pp) else None}
    overall = one(np.ones(len(y), bool)); axes = {}
    for axis in AXES:
        labels = np.asarray([r[axis] for r in rows]); groups = {str(g): one(labels == g) for g in np.unique(labels)}
        names = eligible[axis] if eligible is not None else [g for g, v in groups.items() if g.lower() != 'unknown' and min(v['n_real'], v['n_fake']) >= min_support]
        good = [groups[g] for g in names if g in groups]
        valid = len(good) == len(names) and len(good) >= 2 and all(r['auc'] is not None for r in good) and overall['auc'] is not None
        axes[axis] = {'groups': groups, 'eligible_groups': list(names), 'min_support_per_class': min_support,
                      'worst_group_auc': min(r['auc'] for r in good) if good and all(r['auc'] is not None for r in good) else None}
        for key in ('fpr', 'fnr', 'auc'):
            axes[axis][key + '_gap'] = max(r[key] for r in good) - min(r[key] for r in good) if valid else None
        axes[axis]['F_FPR'] = sum(abs(r['fpr'] - overall['fpr']) for r in good) if valid else None
        axes[axis]['F_DP'] = max(r['positive_rate'] for r in good) - min(r['positive_rate'] for r in good) if valid else None
        axes[axis]['es_AUC'] = overall['auc'] / (1 + sum(abs(r['auc'] - overall['auc']) for r in good)) if valid else None
    families = {}
    for family in sorted({r['family'] for r in rows if r['label'] == 1}):
        mask = np.asarray([r['label'] == 0 or r['family'] == family for r in rows])
        families[family] = one(mask)
    return {'units': 'fractions', 'threshold_logit': float(threshold), 'overall': overall, 'axes': axes, 'by_fake_family_vs_all_reals': families}


def check_config(c, fixture=False, bound=True):
    keys = {'format', 'dataset_id', 'train', 'val', 'train_sha256', 'val_sha256', 'overlap_file',
            'overlap_sha256', 'init_checkpoint', 'init_sha256', 'seed', 'hidden', 'gate_hidden',
            'size', 'epochs', 'batch_size', 'workers', 'encoder_policy', 'encoder_lr', 'head_lr',
            'weight_decay', 'axes', 'balance_axis', 'min_train_support', 'min_eval_support',
            'group_per_class', 'aux_every', 'train_calibration_size', 'target_fpr', 'temperature',
            'margin', 'lambda_fair', 'lambda_sensitivity', 'lambda_alignment', 'sham_weight', 'amp',
            'max_gpu_hours', 'allowed_content_overlap', 'source_sha256', 'synthetic_fixture',
            'error_control', 'alignment_warmup_epochs', 'head_type', 'backbone',
            'normalization_mean', 'normalization_std', 'nuisance_mode'}
    if set(c) != keys: raise ValueError(f'Config schema differs: missing={keys-set(c)}, extra={set(c)-keys}')
    if c['nuisance_mode'] not in ('none', 'matched'):
        raise ValueError('Invalid nuisance mode')
    if c['head_type'] != 'linear' or c['backbone'] != 'resnet50':
        raise ValueError('Table 3 adapter is fixed to ResNet-50 plus one linear head')
    if c['normalization_mean'] != [0.485, 0.456, 0.406] or c['normalization_std'] != [0.229, 0.224, 0.225]:
        raise ValueError('ResNet ImageNet normalization changed')
    if c['error_control'] not in ('legacy', 'support_upper', 'hierarchical_real'):
        raise ValueError('Invalid error-control objective')
    if c['error_control'] == 'hierarchical_real':
        if c['alignment_warmup_epochs'] != 0 or 'gender_race' not in c['axes']:
            raise ValueError('Hierarchical real-only candidate requires joint batches and no ramp')
        if not fixture and c['axes'] != ['gender', 'race', 'age_bin', 'skin_bin', 'gender_race']:
            raise ValueError('Do not change the D auxiliary schedule')
    if type(c['alignment_warmup_epochs']) is not int or not 0 <= c['alignment_warmup_epochs'] <= c['epochs']:
        raise ValueError('Invalid alignment warmup')
    if c['format'] != VERSION or bool(c['synthetic_fixture']) != fixture: raise ValueError('Config format/fixture mismatch')
    for key in ('seed', 'hidden', 'gate_hidden', 'size', 'epochs', 'batch_size', 'workers', 'min_train_support',
                'min_eval_support', 'group_per_class', 'aux_every', 'train_calibration_size'):
        if type(c[key]) is not int or c[key] < (0 if key in ('seed', 'workers') else 1): raise ValueError(f'Invalid {key}')
    if c['size'] < 71 or (not fixture and c['size'] != 224): raise ValueError('Production preprocessing is fixed224')
    if not fixture and (c['hidden'], c['gate_hidden']) != (128, 64): raise ValueError('Production head widths fixed')
    for key in ('encoder_lr', 'head_lr', 'weight_decay', 'temperature', 'margin', 'lambda_fair', 'lambda_sensitivity', 'lambda_alignment', 'sham_weight', 'max_gpu_hours'):
        if not isinstance(c[key], (float, int)) or not math.isfinite(c[key]) or c[key] <= 0: raise ValueError(f'Invalid {key}')
    if type(c['amp']) is not bool or not 0 < c['target_fpr'] < 1: raise ValueError('Invalid AMP/FPR')
    if c['balance_axis'] not in AXES or c['encoder_policy'] not in ('frozen', 'exit', 'full'): raise ValueError('Invalid model/sampling policy')
    if not c['axes'] or len(set(c['axes'])) != len(c['axes']) or not set(c['axes']).issubset(AXES): raise ValueError('Invalid axes')
    if not isinstance(c['allowed_content_overlap'], list): raise ValueError('Content exception must be explicit hash list')
    for key, hkey in (('train', 'train_sha256'), ('val', 'val_sha256'), ('init_checkpoint', 'init_sha256')):
        if sha(c[key]) != c[hkey]: raise ValueError(f'{key} hash mismatch')
    if bool(c['overlap_file']) != bool(c['overlap_sha256']): raise ValueError('Incomplete overlap binding')
    if c['overlap_file'] and sha(c['overlap_file']) != c['overlap_sha256']: raise ValueError('Overlap exception changed')
    if bound and c['source_sha256'] != sha(__file__): raise ValueError('Source changed; old checks cannot certify it')
    return c


def load_config(path, fixture=False):
    return check_config(json.loads(Path(path).read_text()), fixture=fixture)


def bind(template, out):
    c = json.loads(Path(template).read_text())
    c['source_sha256'] = sha(__file__)
    check_config(c)
    if Path(out).exists(): raise ValueError('Do not overwrite bound config')
    model = load_initial(c)
    write_json(out, c)
    print(json.dumps({'status': 'BOUND', 'config_sha256': digest(c), 'parameters': model.counts()}))


def preflight(c, out):
    out = fresh_dir(out); train, val = read_manifest(c['train']), read_manifest(c['val'])
    audit = split_audit(train, val, c['overlap_file'])
    _, support = group_inventory(train, c)
    inventory = []
    hashes_by_split = {}
    for split, rows in (('train', train), ('val', val)):
        hashes_by_split[split] = set()
        for r in rows:
            h = sha(r['image']); hashes_by_split[split].add(h)
            # Decode every original, without inference or TEST access.
            with Image.open(r['image']) as im:
                im.convert('RGB').load()
                if min(im.size) < 2: raise ValueError('Invalid image dimensions')
            inventory.append({'split': split, 'image': r['image'], 'sha256': h})
    overlap = sorted(hashes_by_split['train'] & hashes_by_split['val'])
    if overlap != sorted(c['allowed_content_overlap']): raise ValueError('Undeclared/changed cross-split byte overlap; preserve split, review exception explicitly')
    write_json(out / 'inventory_PRIVATE.json', inventory)
    record = {'status': 'PASS', 'source_sha256': sha(__file__), 'config_digest': digest(c),
              'inventory_sha256': sha(out / 'inventory_PRIVATE.json'), 'decoded_images': len(inventory),
              'train_rows': len(train), 'val_rows': len(val), 'split_audit': audit, 'groups': support,
              'cross_split_content_overlap': len(overlap), 'test_read': False, 'synthetic_fixture': c['synthetic_fixture']}
    write_json(out / 'PREFLIGHT.json', record)
    print(json.dumps({'status': 'PREFLIGHT_PASS', 'train': len(train), 'val': len(val)}))
    return record


def checked_inputs(c, preflight_dir):
    root = Path(preflight_dir); receipt = json.loads((root / 'PREFLIGHT.json').read_text())
    if receipt['status'] != 'PASS' or receipt['config_digest'] != digest(c) or receipt['source_sha256'] != sha(__file__):
        raise ValueError('Stale input checks')
    if receipt['inventory_sha256'] != sha(root / 'inventory_PRIVATE.json'): raise ValueError('Preflight inventory changed')
    inv = json.loads((root / 'inventory_PRIVATE.json').read_text())
    look = {(r['split'], r['image']): r['sha256'] for r in inv}
    train, val = read_manifest(c['train']), read_manifest(c['val'])
    for split, rows in (('train', train), ('val', val)):
        for r in rows: r['image_sha256'] = look[(split, r['image'])]
    return train, val, receipt


def image_uint8(x): return (x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def preview(c, preflight_dir, out):
    out = fresh_dir(out); train, _, _ = checked_inputs(c, preflight_dir); active, _ = group_inventory(train, c)
    chosen = set()
    for axis, groups in active.items():
        for group, cell in groups.items():
            chosen.add(min(cell[0], key=lambda ix: seed_for(c['seed'], axis, group, train[ix]['image'])))
    records = []
    for ix in sorted(chosen):
        x = load_pixels(train[ix], c['size'])
        for k, op in enumerate(OPERATORS):
            for j, strength in enumerate(STRENGTHS):
                s, t, mask, settings = intervention(x, op, strength, seed_for(c['seed'], ix, k, j, 'review'))
                cases = NUISANCE_BANK if c['nuisance_mode'] == 'matched' else ('identity',)
                for nu in cases:
                    nx, ns, nt = common_nuisance(x, s, t, nu)
                    name = f'case_{ix:06d}_{k}_{j}_{nu}'
                    panels = (nx, ns, nt, (nt - ns).abs() * 4)
                    sheet = Image.new('RGB', (c['size'] * 4, c['size'] + 24), 'white')
                    draw = ImageDraw.Draw(sheet)
                    for p, (label, panel) in enumerate(zip(('original: '+nu, 'processing control', 'local intervention', 'difference x4'), panels)):
                        sheet.paste(Image.fromarray(image_uint8(panel)), (p * c['size'], 24))
                        draw.text((p * c['size'] + 4, 4), label, fill='black')
                    path = out / (name + '.png'); sheet.save(path)
                    records.append({'id': name, 'image': train[ix]['image'], 'image_sha256': train[ix]['image_sha256'],
                                    'sheet': str(path.resolve()), 'sheet_sha256': sha(path),
                                    'settings': {**settings, 'common_post_nuisance': nu},
                                    'groups': {a: train[ix][a] for a in c['axes']},
                                    'mask_area_fraction': float(mask.mean()),
                                    'pixel_pair_difference_before': float((t-s).abs().mean()),
                                    'pixel_pair_difference_after': float((nt-ns).abs().mean())})
    manifest = {'source_sha256': sha(__file__), 'config_digest': digest(c), 'synthetic_fixture': c['synthetic_fixture'], 'cases': records}
    write_json(out / 'preview_PRIVATE.json', manifest)
    write_json(out / 'review_PENDING_PRIVATE.json', {'preview_sha256': sha(out / 'preview_PRIVATE.json'), 'cases': [
        {'id': r['id'], 'approved': False, 'reviewer': '', 'face_crop_suitable': False,
         'control_processing_acceptable': False, 'intended_local_change_present': False,
         'demographic_appearance_preserved': False, 'evidence_notes': ''} for r in records]})
    print(json.dumps({'status': 'REVIEW_PENDING', 'cases': len(records), 'review': str(out / 'review_PENDING_PRIVATE.json')}))


def qualify(preview_dir, review_file, out):
    root = Path(preview_dir); pv = root / 'preview_PRIVATE.json'; manifest = json.loads(pv.read_text())
    review = json.loads(Path(review_file).read_text())
    if manifest['synthetic_fixture']: raise ValueError('Fixture review cannot qualify production interventions')
    if manifest['source_sha256'] != sha(__file__) or review['preview_sha256'] != sha(pv): raise ValueError('Stale operator review')
    expected = {r['id']: r for r in manifest['cases']}; checked = set(); reviewers = set()
    flags = ('approved', 'face_crop_suitable', 'control_processing_acceptable', 'intended_local_change_present', 'demographic_appearance_preserved')
    for r in review['cases']:
        if r['id'] in checked or r['id'] not in expected: raise ValueError('Review IDs invalid')
        if any(r.get(k) is not True for k in flags) or not r.get('reviewer', '').strip() or not r.get('evidence_notes', '').strip():
            raise ValueError('Substantive named operator review incomplete/rejected; revise recipe under new proposal')
        item = expected[r['id']]
        if sha(item['sheet']) != item['sheet_sha256'] or sha(item['image']) != item['image_sha256']: raise ValueError('Reviewed bytes changed')
        checked.add(r['id']); reviewers.add(r['reviewer'])
    if checked != set(expected): raise ValueError('Missing operator review cases')
    if Path(out).exists(): raise ValueError('Do not overwrite qualification')
    write_json(out, {'status': 'OPERATOR_REVIEW_RECORDED', 'source_sha256': sha(__file__),
                    'config_digest': manifest['config_digest'], 'preview_sha256': sha(pv),
                    'review_sha256': sha(review_file), 'preview_file': str(pv.resolve()),
                    'review_file': str(Path(review_file).resolve()), 'case_count': len(checked),
                    'reviewers': sorted(reviewers), 'real_forgery_validity_proven': False,
                    'scope': 'Stratified visual qualification of defined procedural probes; not per-image biological identity or causal validation'})


def check_qualification(c, path):
    if not path: raise ValueError('C/D require recorded operator review; use preview then qualify')
    q = json.loads(Path(path).read_text())
    if q['status'] != 'OPERATOR_REVIEW_RECORDED' or q['source_sha256'] != sha(__file__) or q['config_digest'] != digest(c): raise ValueError('Qualification mismatch')
    if sha(q['preview_file']) != q['preview_sha256'] or sha(q['review_file']) != q['review_sha256']: raise ValueError('Qualification evidence changed')
    return sha(path)


def build_optimizer(model, c):
    encoder, head = [], []
    for name, p in model.named_parameters():
        if p.requires_grad: (encoder if name.startswith('encoder.') else head).append(p)
    groups = [{'params': head, 'lr': c['head_lr']}]
    if encoder: groups.append({'params': encoder, 'lr': c['encoder_lr']})
    return torch.optim.AdamW(groups, weight_decay=c['weight_decay'])


def finite_step(losses, model, opt):
    if not all(torch.isfinite(v) for v in losses.values()): raise RuntimeError('Nonfinite loss')
    opt.zero_grad(set_to_none=True); losses['total'].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    if any(g is None or not torch.isfinite(g).all() for g in grads): raise RuntimeError('Missing/nonfinite gradient')
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
    opt.step()


def train(c, preflight_dir, out, arm, device, qualification=None, resume=False, fixture=False, stop_after=None, budget_hours=None):
    if arm not in ARMS: raise ValueError(arm)
    budget_hours = c['max_gpu_hours'] if budget_hours is None else budget_hours
    if not math.isfinite(budget_hours) or budget_hours <= 0: raise ValueError('Invalid execution budget')
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if device.type == 'cuda' and c['amp'] and not torch.cuda.is_bf16_supported(): raise RuntimeError('Configured bf16 unsupported')
    qhash = check_qualification(c, qualification) if arm in ('C', 'D') and not fixture else None
    rows, val, pre = checked_inputs(c, preflight_dir); active, support = group_inventory(rows, c)
    out = Path(out); seed_all(c['seed']); model = load_initial(c).to(device); opt = build_optimizer(model, c)
    weights = original_weights(rows, c['balance_axis']); best = -math.inf; start_epoch = 0; elapsed = 0.; history = []
    real_rows = [r for r in rows if r['label'] == 0]
    real_rows = sorted(real_rows, key=lambda r: seed_for(c['seed'], 'train-calibration', r['image']))[:c['train_calibration_size']]
    if resume:
        if (out / 'DONE').exists(): raise ValueError('Completed run preserved')
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        for key, value in (('config', c), ('source_sha256', sha(__file__)), ('arm', arm), ('qualification_sha256', qhash)):
            if ck.get(key) != value: raise ValueError(f'Resume mismatch: {key}')
        model.load_state_dict(ck['model'], strict=True); opt.load_state_dict(ck['optimizer'])
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value): state[key] = value.to(device)
        start_epoch, best, elapsed, history = ck['epoch'], ck['best_auc'], ck['elapsed_seconds'], ck['history']
        if not (out / 'best.pt').exists() or sha(out / 'best.pt') != ck['best_checkpoint_sha256']: raise ValueError('Selected checkpoint missing/changed')
    else:
        fresh_dir(out); write_json(out / 'config.json', c)
        write_json(out / 'RUN.json', {'format': VERSION, 'arm': arm, 'source_sha256': sha(__file__), 'config_digest': digest(c),
                                    'parameters': model.counts(), 'groups': support, 'qualification_sha256': qhash,
                                    'input_receipt': pre, 'device': str(device), 'torch': str(torch.__version__), 'test_read': False})
    with (out / 'execution_attempts.jsonl').open('a') as f:
        f.write(json.dumps({'resume': resume, 'cumulative_budget_hours': budget_hours, 'completed_epochs_before': start_epoch,
                            'source_sha256': sha(__file__), 'scientific_config_digest': digest(c)}) + '\n')
    begin = time.perf_counter(); base_elapsed = elapsed
    for epoch in range(start_epoch, c['epochs']):
        elapsed = base_elapsed + time.perf_counter() - begin
        estimate = history[-1]['epoch_seconds'] if history else 0.
        if elapsed + estimate > budget_hours * 3600 or (stop_after is not None and epoch >= stop_after):
            write_json(out / 'STATUS.json', {'status': 'INCOMPLETE_EPOCH_BOUNDARY_STOP', 'completed_epochs': epoch, 'planned_epochs': c['epochs']})
            return {'status': 'INCOMPLETE', 'epochs': epoch}
        epoch_start = time.perf_counter(); seed_all(c['seed'] + epoch)
        # Calibration uses a fixed, naturally sampled TRAIN-real subset, never VAL.
        threshold_record = calibrate(predict(model, real_rows, c, device), c['target_fpr'])
        model.train(); seen = Counter(); paired = 0; sums = Counter(); aux_events = Counter(); aux_draws = Counter(); steps = 0
        original_pair_groups = defaultdict(Counter); auxiliary_cells = Counter(); support_histogram = Counter()
        nuisance_counts = Counter(); nuisance_by_group_class = Counter(); aux_nuisance_counts = Counter()
        auxiliary_event = epoch * math.ceil(math.ceil(len(rows) / c['batch_size']) / c['aux_every'])
        for step, b in enumerate(make_loader(rows, c, epoch, pairs=arm in ('C', 'D'), shuffle=True)):
            aux = None
            if step % c['aux_every'] == 0:
                # Deterministic auxiliary plan identical across arms. A does not forward it.
                aux = auxiliary(rows, active, c, auxiliary_event, epoch); auxiliary_event += 1
                aux_events[f"{aux['axis']}:{aux['operator']}:{aux['strength']}"] += 1
                aux_draws.update(aux['indices'])
            losses, n_pair = train_step(model, b, aux, weights, c, arm, threshold_record['threshold_logit'], device, epoch)
            finite_step(losses, model, opt); seen.update(b['index'].tolist()); paired += n_pair; steps += 1
            for ix, nu in zip(b['index'].tolist(), b['nuisance_index'].tolist()):
                case_name = NUISANCE_BANK[nu]; nuisance_counts[case_name] += 1
                for axis in c['axes']:
                    nuisance_by_group_class[f'{axis}:{rows[ix][axis]}:{rows[ix]["label"]}:{case_name}'] += 1
            if aux is not None and arm != 'A':
                aux_nuisance_counts[f'{aux["axis"]}:{aux["nuisance_case"]}'] += 1
            if arm in ('C', 'D'):
                for ix in b['index'].tolist():
                    if rows[ix]['label'] == 0:
                        for axis in c['axes']: original_pair_groups[axis][rows[ix][axis]] += 1
            if aux is not None and arm != 'A':
                for label, group in zip(aux['y'].tolist(), aux['group'].tolist()):
                    auxiliary_cells[f"{aux['axis']}:{aux['groups'][group]}:{label}"] += 1
                for gi, name in enumerate(aux['groups']):
                    for label in (0, 1):
                        fraction = float(aux['support_fraction'][gi, label])
                        support_histogram[f"{aux['axis']}:{name}:{label}:U/n={fraction:.4f}"] += 1
            for key, value in losses.items(): sums[key] += float(value.detach())
            if steps % 100 == 0: print(json.dumps({'arm': arm, 'epoch': epoch + 1, 'step': steps, 'loss': float(losses['total'].detach())}), flush=True)
        if len(seen) != len(rows) or any(v != 1 for v in seen.values()): raise RuntimeError('Original coverage error')
        if arm in ('C', 'D') and paired != sum(r['label'] == 0 for r in rows): raise RuntimeError('Real-source intervention coverage error')
        zv = predict(model, val, c, device); yc = np.asarray([r['label'] for r in val])
        cal = calibrate(zv[yc == 0], c['target_fpr']); report = metric_report(val, zv, cal['threshold_logit'], c['min_eval_support'])
        auc = report['overall']['auc']; improved = auc > best
        if improved: best = auc
        record = {'epoch': epoch + 1, 'steps': steps, 'losses_step_mean': {k: v / steps for k, v in sums.items()},
                  'unique_originals': len(seen), 'originals_once_each': True, 'original_real_pairs': paired,
                  'auxiliary_schedule': dict(aux_events), 'auxiliary_planned_draws': sum(aux_draws.values()),
                  'auxiliary_forwarded_originals': sum(aux_draws.values()) if arm != 'A' else 0,
                  'auxiliary_unique_planned_sources': len(aux_draws),
                  'original_real_pairs_by_axis_group': {a: dict(v) for a, v in original_pair_groups.items()},
                  'auxiliary_original_draws_by_axis_group_label': dict(auxiliary_cells),
                  'auxiliary_real_pair_draws_by_axis_group': {k: v for k, v in auxiliary_cells.items() if k.endswith(':0')} if arm in ('C', 'D') else {},
                  'recorded_contributor_support_histogram': dict(support_histogram),
                  'alignment_multiplier': alignment_multiplier(epoch, c['alignment_warmup_epochs']),
                  'main_nuisance_counts': dict(nuisance_counts),
                  'main_nuisance_by_axis_group_class': dict(nuisance_by_group_class),
                  'aux_nuisance_events': dict(aux_nuisance_counts),
                  'train_calibration': threshold_record, 'validation': report, 'calibration': cal,
                  'selected_this_epoch': improved, 'epoch_seconds': time.perf_counter() - epoch_start}
        history.append(record); elapsed = base_elapsed + time.perf_counter() - begin
        ck = {'format': VERSION, 'arm': arm, 'config': c, 'source_sha256': sha(__file__), 'model': model.state_dict(),
              'optimizer': opt.state_dict(), 'epoch': epoch + 1, 'best_auc': best, 'validation': report, 'calibration': cal,
              'elapsed_seconds': elapsed, 'history': history, 'qualification_sha256': qhash,
              'synthetic_fixture': fixture, 'fit_membership': [{k: r[k] for k in ('image', 'identity_ids', 'source_ids', 'image_sha256')} for r in rows + val]}
        if improved: atomic_save(out / 'best.pt', ck)
        ck['best_checkpoint_sha256'] = sha(out / 'best.pt'); atomic_save(out / 'last.pt', ck)
        write_json(out / 'history.json', history)
        print(json.dumps({'arm': arm, 'epoch': epoch + 1, 'validation_auc': auc, 'paired_sources': paired, 'seconds': elapsed}), flush=True)
    # Independent reload and native validation replay certify the selected artifact.
    selected = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    replay_model = Detector(c['hidden'], c['gate_hidden'], c['encoder_policy'], c['head_type']).to(device)
    replay_model.load_state_dict(selected['model'], strict=True)
    replay_logits = predict(replay_model, val, c, device)
    replay = metric_report(val, replay_logits, selected['calibration']['threshold_logit'], c['min_eval_support'])
    if replay != selected['validation']: raise RuntimeError('Selected checkpoint validation replay mismatch')
    # Fixed VAL predictions permit paired uncertainty without more GPU inference.
    # This artifact is private and cannot authorize any TEST access.
    write_json(out / 'validation_scores_PRIVATE.json', {
        'rows': val, 'logits': replay_logits.tolist(), 'threshold': selected['calibration']['threshold_logit'],
        'min_support': c['min_eval_support'], 'manifest_sha256': c['val_sha256'],
        'checkpoint_sha256': sha(out / 'best.pt'), 'calibration_source': 'selected checkpoint validation only',
        'split': 'validation', 'test_read': False})
    result = {'status': 'COMPLETED_VERIFIED', 'arm': arm, 'epochs_completed': c['epochs'], 'selected_epoch': selected['epoch'],
              'selected_checkpoint_sha256': sha(out / 'best.pt'), 'source_sha256': sha(__file__), 'config_digest': digest(c),
              'validation': replay, 'test_evaluated': False, 'elapsed_seconds': elapsed, 'synthetic_fixture': fixture}
    write_json(out / 'RUN_VERIFIED.json', result); (out / 'DONE').write_text('Completed; selected checkpoint independently replayed on VAL. TEST not read.\n')
    return result


def dry_run(c, preflight_dir, out, arm, device):
    out = fresh_dir(out); rows, _, _ = checked_inputs(c, preflight_dir); active, _ = group_inventory(rows, c)
    model = load_initial(c).to(device); opt = build_optimizer(model, c); weights = original_weights(rows, c['balance_axis'])
    # Ensure an actual real input is represented, without changing production sampling.
    ix = next(i for i, r in enumerate(rows) if r['label'] == 0)
    samples = [Originals(rows, c, pairs=True, training=True)[i] for i in [ix] + [j for j in range(min(c['batch_size'] - 1, len(rows)))]]
    b = {k: torch.stack([s[k] for s in samples]) if torch.is_tensor(samples[0][k]) else torch.tensor([s[k] for s in samples]) for k in samples[0]}
    event = c['axes'].index('gender_race') if c['error_control'] == 'hierarchical_real' else 0
    aux = auxiliary(rows, active, c, event, 0)
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    real_rows = sorted([r for r in rows if r['label'] == 0],
                       key=lambda r: seed_for(c['seed'], 'train-calibration', r['image']))[:c['train_calibration_size']]
    threshold_record = calibrate(predict(model, real_rows, c, device), c['target_fpr'])
    model.train()
    started = time.perf_counter()
    losses, count = train_step(model, b, aux, weights, c, arm, threshold_record['threshold_logit'], device)
    changed_grad_norm = None
    if c['error_control'] == 'hierarchical_real' and arm != 'A':
        grads = torch.autograd.grad(losses['error_fairness'], [p for p in model.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)
        changed_grad_norm = math.sqrt(sum(float(g.float().square().sum()) for g in grads if g is not None))
        if not math.isfinite(changed_grad_norm) or changed_grad_norm <= 0:
            raise RuntimeError('Changed real-input joint error loss has no finite gradient')
    finite_step(losses, model, opt)
    if device.type == 'cuda': torch.cuda.synchronize(device)
    record = {'status': 'PASS', 'source_sha256': sha(__file__), 'config_digest': digest(c), 'arm': arm,
              'error_control': c['error_control'], 'alignment_multiplier': alignment_multiplier(0, c['alignment_warmup_epochs']),
              'parameters': model.parameter_report(), 'head_type': model.head_type,
              'train_calibration': threshold_record, 'seconds_one_optimizer_step': time.perf_counter() - started,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'losses': {k: float(v.detach()) for k, v in losses.items()}, 'real_pair_count': count,
              'optimizer_steps_in_discarded_model': 1, 'scientific_intervention_validity': False,
              'auxiliary_axis_checked': aux['axis'], 'changed_error_gradient_norm': changed_grad_norm,
              'device': str(device), 'torch': str(torch.__version__)}
    write_json(out / 'DRY_RUN.json', record); print(json.dumps(record))
    return record


def flat_metrics(report):
    out = {f'overall/{k}': report['overall'][k] for k in ('auc', 'fpr', 'fnr')}
    for axis, data in report['axes'].items():
        for key in ('F_FPR', 'F_DP', 'es_AUC', 'worst_group_auc', 'fpr_gap', 'fnr_gap'):
            out[f'{axis}/{key}'] = data[key]
        for name, group in data['groups'].items():
            for key in ('auc', 'fpr', 'fnr'):
                out[f'{axis}/group={name}/{key}'] = group[key]
    return out


def components(rows):
    parent = list(range(len(rows))); seen = {}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, row in enumerate(rows):
        tokens = [('image', row['image'])]
        for field in ('identity_ids', 'source_ids'): tokens += [(field, v) for v in ids(row[field])]
        for token in tokens:
            if token in seen: parent[root(i)] = root(seen[token])
            else: seen[token] = i
    groups = defaultdict(list)
    for i in range(len(rows)): groups[root(i)].append(i)
    return [np.asarray(v, int) for v in groups.values()]


def bootstrap(rows, z, threshold, min_support, n, seed, other=None):
    """Connected recorded-contributor resampling; same draws for paired differences.
    Fixed original eligibility, frozen thresholds. No training-seed uncertainty.
    """
    reference = metric_report(rows, z, threshold, min_support)
    eligible = {a: r['eligible_groups'] for a, r in reference['axes'].items()}
    points = flat_metrics(reference); groups = components(rows); values = defaultdict(list)
    if other is not None:
        other_z, other_threshold = other
        other_points = flat_metrics(metric_report(rows, other_z, other_threshold, min_support, eligible))
        points = {k: points[k] - other_points[k] if points[k] is not None and other_points.get(k) is not None else None for k in points}
    rng = np.random.default_rng(seed)
    for _ in range(n if len(groups) >= 2 else 0):
        ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        rr = [rows[i] for i in ix]
        current = flat_metrics(metric_report(rr, np.asarray(z)[ix], threshold, min_support, eligible))
        alternate = flat_metrics(metric_report(rr, np.asarray(other_z)[ix], other_threshold, min_support, eligible)) if other is not None else None
        for k in points:
            value = current.get(k)
            if alternate is not None:
                value = value - alternate[k] if value is not None and alternate.get(k) is not None else None
            if value is not None and math.isfinite(value): values[k].append(value)
    intervals = {}
    for k, point in points.items():
        v = values[k]
        intervals[k] = {'estimate': point, 'ci95': np.quantile(v, [.025, .975]).tolist() if len(v) >= 2 else None,
                        'valid_replicates': len(v), 'invalid_or_unavailable_replicates': n - len(v)}
    return {'method': 'percentile connected-recorded-contributor bootstrap', 'replicates': n, 'seed': seed,
            'components': len(groups), 'largest_component_rows': max(map(len, groups)),
            'paired_difference': 'first minus second' if other is not None else None,
            'training_seed_variability_measured': False, 'true_person_independence_verified': False,
            'eligible_groups_fixed_before_resampling': eligible, 'intervals': intervals}


def load_trained(checkpoint, device, fixture=False):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if ck.get('format') != VERSION or ck.get('source_sha256') != sha(__file__) or bool(ck.get('synthetic_fixture')) != fixture:
        raise ValueError('Checkpoint format/source/fixture mismatch')
    c = ck['config']; model = Detector(c['hidden'], c['gate_hidden'], c['encoder_policy'], c['head_type'])
    model.load_state_dict(ck['model'], strict=True)
    return model.to(device).eval(), ck


def evaluate(checkpoint, manifest, manifest_sha256, out, device, overlap_file=None, replicates=1000, fixture=False):
    if sha(manifest) != manifest_sha256: raise ValueError('Evaluation manifest hash mismatch')
    model, ck = load_trained(checkpoint, device, fixture); c = ck['config']
    rows = read_manifest(manifest, allow_duplicates=True)
    audit = split_audit(ck['fit_membership'], rows, overlap_file, evaluation=True)
    for row in rows: row['image_sha256'] = sha(row['image'])
    content_overlap = sorted({r['image_sha256'] for r in rows} & {r['image_sha256'] for r in ck['fit_membership']})
    declared_content = json.loads(Path(overlap_file).read_text()).get('declared_content_overlap_sha256', []) if overlap_file else []
    if content_overlap != sorted(declared_content): raise ValueError('Undeclared/changed fit:TEST byte overlap')
    audit['content_overlap_count'] = len(content_overlap)
    out = fresh_dir(out); before = sha(checkpoint)
    z = predict(model, rows, c, device); threshold = ck['calibration']['threshold_logit']
    report = {'checkpoint_sha256': before, 'manifest_sha256': manifest_sha256, 'source_sha256': sha(__file__),
              'calibration': ck['calibration'], 'split_audit': audit,
              'image_inventory_digest': digest([(r['image'], r['image_sha256']) for r in rows]),
              'calibrated': metric_report(rows, z, threshold, c['min_eval_support']),
              'fixed_probability_0_5': metric_report(rows, z, 0., c['min_eval_support']),
              'uncertainty': bootstrap(rows, z, threshold, c['min_eval_support'], replicates, c['seed'])}
    write_json(out / 'metrics.json', report)
    # Private row-level artifact needed for component/paired reanalysis; never push.
    write_json(out / 'scores_PRIVATE.json', {'rows': rows, 'logits': z.tolist(), 'threshold': threshold,
                                          'min_support': c['min_eval_support'], 'manifest_sha256': manifest_sha256,
                                          'checkpoint_sha256': before, 'calibration_source': 'selected checkpoint validation only'})
    if before != sha(checkpoint): raise RuntimeError('Checkpoint changed during evaluation')
    return report


def compare(first, second, out, replicates, seed):
    a, b = json.loads(Path(first).read_text()), json.loads(Path(second).read_text())
    if a['rows'] != b['rows'] or a['manifest_sha256'] != b['manifest_sha256'] or a['min_support'] != b['min_support']:
        raise ValueError('Paired comparison requires identical row order, membership, annotations and eligibility')
    if Path(out).exists(): raise ValueError('Preserve prior comparison')
    write_json(out, bootstrap(a['rows'], a['logits'], a['threshold'], a['min_support'], replicates, seed,
                              other=(b['logits'], b['threshold'])))


@torch.inference_mode()
def export(checkpoint, images, out, device, fixture=False):
    model, ck = load_trained(checkpoint, device, fixture); out = fresh_dir(out); records = []
    for i, path in enumerate(images):
        raw_image = load_pixels({'image': path}, ck['config']['size'])
        r = model(normalize(raw_image)[None].to(device))
        local = r['local_map'][0].float().cpu().numpy()
        up = F.interpolate(r['local_map'][:, None].float(), size=raw_image.shape[-2:], mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        scale = max(float(np.abs(up).max()), 1e-12); t = np.clip(up / scale, -1, 1)
        heat = np.stack((1 - np.maximum(-t, 0), 1 - abs(t), 1 - np.maximum(t, 0)), -1)
        name = f'{i:04d}'
        Image.blend(Image.fromarray(image_uint8(raw_image)), Image.fromarray(np.rint(heat * 255).astype(np.uint8)), .45).save(out / (name + '_local.png'))
        arrays = {'local_map': local}
        if 'patch_gate' in r: arrays['patch_gate'] = r['patch_gate'][0].cpu().numpy()
        np.savez_compressed(out / (name + '_raw.npz'), **arrays)
        z = float(r['logit'][0]); global_part = float(r['global_contribution'][0]); bias = float(r['bias'])
        if not math.isclose(z, global_part + float(local.mean()) + bias, abs_tol=2e-5, rel_tol=2e-5): raise RuntimeError('Map decomposition failed')
        records.append({'image': str(Path(path).resolve()), 'image_sha256': sha(path), 'logit': z,
                        'prediction': int(z >= ck['calibration']['threshold_logit']), 'global': global_part,
                        'local_mean': float(local.mean()), 'bias': bias, 'display_abs_scale': scale,
                        'head_type': model.head_type})
    write_json(out / 'scores_PRIVATE.json', records)
    scope = ('Linear CAM: map mean plus classifier bias equals the complete logit; no gates or separate global branch. '
             if model.head_type == 'linear' else 'Gated LOCAL contribution only; global contribution and bias are separate. ')
    (out / 'README.txt').write_text(scope + 'Red/blue are positive/negative signed contributions, not validated forgery masks. Per-image scale; use raw values for comparison.\n')


def defaults():
    return {'format': VERSION, 'dataset_id': 'REPLACE', 'train': '', 'val': '', 'train_sha256': '', 'val_sha256': '',
            'overlap_file': None, 'overlap_sha256': None, 'init_checkpoint': '', 'init_sha256': '',
            'seed': 17, 'hidden': 128, 'gate_hidden': 64, 'size': 224, 'epochs': 10, 'batch_size': 32, 'workers': 4,
            'encoder_policy': 'exit', 'encoder_lr': 1e-5, 'head_lr': 1e-3, 'weight_decay': 1e-4,
            'axes': ['gender', 'race', 'age_bin', 'skin_bin', 'gender_race'], 'balance_axis': 'ids8',
            'min_train_support': 20, 'min_eval_support': 20, 'group_per_class': 4, 'aux_every': 4,
            'train_calibration_size': 2048, 'target_fpr': .1, 'temperature': 1., 'margin': .5,
            'lambda_fair': .1, 'lambda_sensitivity': .1, 'lambda_alignment': .1, 'sham_weight': .5,
            'amp': True, 'max_gpu_hours': 2., 'allowed_content_overlap': [], 'source_sha256': None, 'synthetic_fixture': False,
            'error_control': 'legacy', 'alignment_warmup_epochs': 0, 'head_type': 'linear',
            'backbone': 'resnet50', 'normalization_mean': [0.485, 0.456, 0.406],
            'normalization_std': [0.229, 0.224, 0.225], 'nuisance_mode': 'none'}


def smoke(device, out, head_type='linear', nuisance_mode='none'):
    """Full-width head/backbone checks and tiny synthetic A/H lifecycle; no real data."""
    out = fresh_dir(out); seed_all(17); torch.set_num_threads(2); checks = []
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    full = Detector(policy='exit', head_type=head_type).to(device)
    expected = {'linear': (23510081, 14966785)}[head_type]
    assert (full.counts()['total'], full.counts()['trainable']) == expected
    buffers = {k: v.clone() for k, v in full.named_buffers()}
    xx = torch.randn(2, 3, 224, 224, device=device); o = full(xx)
    torch.testing.assert_close(o['logit'], o['global_contribution'] + o['local_map'].mean((1,2)) + o['bias'])
    if head_type == 'linear':
        assert not any(hasattr(full, k) for k in ('global_head','patch_head','image_gate','patch_gate','fusion_raw','fusion_bias'))
        assert 'patch_gate' not in o and 'image_gate' not in o
        with torch.no_grad():
            reference = full.classifier(full.encoder(xx).mean((2,3))).squeeze(1)
        # Separate grad/no-grad convolution forwards can differ by roundoff.
        torch.testing.assert_close(o['logit'], reference, rtol=1e-5, atol=1e-7)
    F.binary_cross_entropy_with_logits(o['logit'], torch.tensor([0.,1.], device=device)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in full.parameters() if p.requires_grad)
    assert sum(float(p.grad.abs().sum()) for p in full.parameters() if p.requires_grad) > 0
    assert all(p.grad is None for p in full.encoder.parameters() if not p.requires_grad)
    assert all(torch.equal(v,dict(full.named_buffers())[k]) for k,v in buffers.items())
    counts = full.parameter_report(); del full, xx, o, buffers
    checks.append('full_width_224px_parameters_cam_gradients_and_frozen_bn')
    if device.type == 'cuda': torch.cuda.empty_cache()
    fixture = out/'fixture_PRIVATE'; fixture.mkdir(); manifests = {}
    rng = np.random.default_rng(44)
    for split in ('train','val','test'):
        rows = []
        for i in range(8):
            path = fixture/f'{split}_{i}.png'
            Image.fromarray(rng.integers(0,256,(75,75,3),dtype=np.uint8)).save(path)
            j=i//2; gender='A' if j<2 else 'B'; race='X' if j%2==0 else 'Y'
            row={'image':str(path.resolve()),'label':i%2,'identity_ids':f'{split}_{i}',
                 'source_ids':f'{split}_{i}','family':'fixture',**{axis:str(j) for axis in AXES}}
            row.update(gender=gender,race=race,gender_race=gender+'-'+race); rows.append(row)
        manifests[split]=fixture/f'{split}.csv'
        with manifests[split].open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    c=defaults();c.update(nuisance_mode=nuisance_mode,dataset_id='SOFTWARE_FIXTURE',train=str(manifests['train'].resolve()),
        val=str(manifests['val'].resolve()),train_sha256=sha(manifests['train']),val_sha256=sha(manifests['val']),
        hidden=8,gate_hidden=4,size=75,epochs=2,batch_size=4,workers=0,min_train_support=1,min_eval_support=1,
        group_per_class=1,aux_every=1,train_calibration_size=4,amp=False,synthetic_fixture=True,
        source_sha256=sha(__file__),max_gpu_hours=1.,axes=['gender_race'],head_type=head_type,
        encoder_policy='frozen')
    initial=Detector(8,4,'exit',head_type);ip=fixture/'initial.pt'
    atomic_save(ip,{'format':'CESFL_CF5050_v1','stage':'pretrain',
        'config':{'train':c['train'],'val':c['val'],'backbone':'resnet50'},
        'synthetic_fixture':True,'model':initial.state_dict()});del initial
    c['init_checkpoint'],c['init_sha256']=str(ip.resolve()),sha(ip)
    first_initial = load_initial(c).state_dict()
    second_initial = load_initial(c).state_dict()
    assert all(torch.equal(v, second_initial[k]) for k, v in first_initial.items())
    wrong = fixture/'wrong_backbone.pt'
    wrong_payload = torch.load(ip, map_location='cpu', weights_only=True)
    wrong_payload['config'] = {**wrong_payload['config'], 'backbone': 'xception'}
    atomic_save(wrong, wrong_payload)
    wrong_config = {**c, 'init_checkpoint': str(wrong.resolve()), 'init_sha256': sha(wrong)}
    try:
        load_initial(wrong_config)
    except ValueError as exc:
        assert 'ResNet-50 detection pretraining' in str(exc)
    else:
        raise AssertionError('Wrong-backbone checkpoint was accepted')
    check_config(c,fixture=True);preflight(c,out/'preflight_A')
    train(c,out/'preflight_A',out/'A','A',device,fixture=True)
    hc=dict(c,error_control='hierarchical_real')
    check_config(hc,fixture=True);preflight(hc,out/'preflight_H')
    dry_run(hc,out/'preflight_H',out/'dry_H','D',device)
    train(hc,out/'preflight_H',out/'H','D',device,fixture=True)
    train(hc,out/'preflight_H',out/'resume_H','D',device,fixture=True,stop_after=1)
    train(hc,out/'preflight_H',out/'resume_H','D',device,fixture=True,resume=True)
    aa=torch.load(out/'H/last.pt',map_location='cpu',weights_only=True)
    bb=torch.load(out/'resume_H/last.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,bb['model'][k]) for k,v in aa['model'].items())
    assert aa['calibration']==bb['calibration']
    for run in ('A','H'):
        hist=json.loads((out/run/'history.json').read_text())
        assert all(r['originals_once_each'] and r['unique_originals']==8 for r in hist)
        assert all(r['original_real_pairs']==(4 if run=='H' else 0) for r in hist)
    checks.append('frozen_encoder_fixture_identical_initialization_wrong_backbone_rejection_A_and_H_reload_coverage_and_bitwise_resume')
    rep=evaluate(out/'H/best.pt',manifests['test'],sha(manifests['test']),out/'evaluation',device,replicates=10,fixture=True)
    rows=read_manifest(manifests['test']);flipped=[{**r,'label':1-r['label']} for r in rows]
    flip=fixture/'flipped.csv'
    with flip.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flipped[0]));w.writeheader();w.writerows(flipped)
    rep2=evaluate(out/'H/best.pt',flip,sha(flip),out/'evaluation_flipped',device,replicates=0,fixture=True)
    assert rep['calibration']==rep2['calibration']
    compare(out/'H/validation_scores_PRIVATE.json',out/'H/validation_scores_PRIVATE.json',out/'paired_self.json',10,17)
    export(out/'H/best.pt',[rows[0]['image']],out/'export',device,fixture=True)
    with np.load(out/'export/0000_raw.npz') as raw:
        assert ('patch_gate' in raw)==(head_type=='gated')
    checks.append('frozen_threshold_label_flip_paired_comparison_and_head_aware_map_export')
    result={'status':'PASS','source_sha256':sha(__file__),'device':str(device),'torch':str(torch.__version__),
        'head_type':head_type,'nuisance_mode':nuisance_mode,'parameters':counts,'checks':checks,'scientific_performance_tested':False,
        'real_interventions_validated':False,'fixture_only':True,'cuda_tested':device.type=='cuda'}
    write_json(out/'SMOKE_RESULT.json',result);print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('bind');p.add_argument('--template',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('preflight');p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('preview');p.add_argument('--config',required=True);p.add_argument('--preflight',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('qualify');p.add_argument('--preview',required=True);p.add_argument('--review',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('smoke');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--out',required=True)
    p.add_argument('--head-type',choices=['linear'],default='linear')
    p.add_argument('--nuisance-mode',choices=['none','matched'],default='none')
    p=sub.add_parser('parameters');p.add_argument('--head-type',choices=['linear'],default='linear')
    p.add_argument('--encoder-policy',choices=['frozen','exit','full'],default='exit')
    for name in ('dry-run','train'):
        p=sub.add_parser(name);p.add_argument('--config',required=True);p.add_argument('--preflight',required=True)
        p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--out',required=True)
        if name=='train':
            p.add_argument('--qualification');p.add_argument('--resume',action='store_true')
            p.add_argument('--dry-run-receipt',required=True);p.add_argument('--software-receipt',required=True)
            p.add_argument('--budget-hours',type=float,help='Explicit cumulative execution allowance; does not change epochs/optimization/selection')
    p=sub.add_parser('evaluate');p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--out',required=True);p.add_argument('--overlap-file')
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p=sub.add_parser('compare');p.add_argument('--first',required=True);p.add_argument('--second',required=True);p.add_argument('--out',required=True)
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--seed',type=int,default=17)
    p=sub.add_parser('export');p.add_argument('--checkpoint',required=True);p.add_argument('--images',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args();torch.set_num_threads(4)
    device=torch.device(args.device) if hasattr(args,'device') else None
    if hasattr(args,'bootstrap') and args.bootstrap<0:parser.error('bootstrap must be nonnegative')
    if args.command=='bind':bind(args.template,args.out)
    elif args.command=='preflight':preflight(load_config(args.config),args.out)
    elif args.command=='preview':preview(load_config(args.config),args.preflight,args.out)
    elif args.command=='qualify':qualify(args.preview,args.review,args.out)
    elif args.command=='smoke':smoke(device,args.out,args.head_type,args.nuisance_mode)
    elif args.command=='parameters':print(json.dumps(Detector(policy=args.encoder_policy,head_type=args.head_type).parameter_report(),indent=2))
    elif args.command=='dry-run':dry_run(load_config(args.config),args.preflight,args.out,args.arm,device)
    elif args.command=='train':
        c=load_config(args.config)
        for path, kind in ((args.dry_run_receipt,'dry'),(args.software_receipt,'software')):
            receipt=json.loads(Path(path).read_text())
            if receipt.get('status')!='PASS' or receipt.get('source_sha256')!=sha(__file__) or receipt.get('device')!=str(device):
                raise ValueError('Missing/stale checks on the target device type')
            if receipt.get('head_type') != c['head_type']:
                raise ValueError('Software/backward receipt is for a different head')
            if kind == 'software' and receipt.get('nuisance_mode') != c['nuisance_mode']:
                raise ValueError('Software receipt is for a different nuisance mode')
            if kind=='dry' and (receipt.get('config_digest')!=digest(c) or receipt.get('arm')!=args.arm):
                raise ValueError('Real-input backward receipt does not match this arm/config')
        train(c,args.preflight,args.out,args.arm,device,args.qualification,args.resume,budget_hours=args.budget_hours)
    elif args.command=='evaluate':evaluate(args.checkpoint,args.manifest,args.manifest_sha256,args.out,device,args.overlap_file,args.bootstrap)
    elif args.command=='compare':compare(args.first,args.second,args.out,args.bootstrap,args.seed)
    elif args.command=='export':export(args.checkpoint,args.images,args.out,device)


if __name__=='__main__':main()
