#!/usr/bin/env python3
"""FF2 common-recipe runner using the existing verified resume/selection engine.

Runtime adapters are installed only in a fresh imported module, never in old
source files or active jobs. GPU commands require explicit device/allocation
receipts in the operator wrapper. UCF and PG are deliberately not BCE fallbacks.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
PROP = ROOT / 'CES_FL/proposals/ff2_expanded_baselines_20261104_r1'
VERSION = 'FF2_EXPANDED_COMMON_v1'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


zoo = module(Path(__file__).with_name('ff2_expanded_models.py'), 'expanded_zoo')
base = module(Path(__file__).with_name('ff2_fairness_baselines.py'), 'expanded_engine')
common = base.common


def hashes():
    paths = [Path(__file__), Path(zoo.__file__), Path(base.__file__), Path(common.__file__),
             Path(base.losses.__file__), PROP / 'sources.lock.json', PROP / 'methods.json']
    return {str(p.resolve().relative_to(ROOT)): common.sha(p) for p in paths}


def load_config(path, allow_unbound=False):
    c = json.loads(Path(path).read_text())
    if c['format'] != VERSION or c['method'] not in zoo.TRAINABLE_METHODS:
        raise ValueError('No executable adapter for this method; inspect its method card')
    if c['source_hashes'] != hashes():
        raise ValueError('Configuration does not bind the current source')
    fixed = dict(seed=17, epochs=50, batch_size=128, validation_batch=128, size=224,
                 lr=.001, momentum=.9, weight_decay=.005, precision='float32', target_fpr=.1, min_support=20)
    if any(c.get(k) != v for k, v in fixed.items()):
        raise ValueError('Common recipe changed; use a new reviewed proposal')
    expected_norm = ([.48145466,.4578275,.40821073], [.26862954,.26130258,.27577711]) if c['method']=='univfd' else \
                    ([.485,.456,.406], [.229,.224,.225]) if c['method'] in ('vit_b16','efficientnetb4') else ([.5]*3,[.5]*3)
    if (c.get('normalization_mean'),c.get('normalization_std')) != expected_norm or c.get('train_bn') != (c['method']!='univfd'):
        raise ValueError('Declared normalization/BN policy differs from this adapter')
    for key in ('train', 'val', 'native_train'):
        if common.sha(c[key]) != c[key + '_sha256']:
            raise ValueError('Changed input: ' + key)
    if not allow_unbound:
        if not c.get('weights') or not c.get('weights_sha256') or common.sha(c['weights']) != c['weights_sha256']:
            raise ValueError('Verified local pretraining weights required')
    return c


def configure(c, resume_steps=0, fixture=False):
    """Reuse original membership/preflight/training engine with explicit adapters."""
    original_images = module(Path(base.__file__), 'expanded_image_source').Images
    mean = torch.tensor(c['normalization_mean'])[:, None, None]
    std = torch.tensor(c['normalization_std'])[:, None, None]

    class Images(original_images):
        def __getitem__(self, index):
            x, y, g, i = super().__getitem__(index)
            return (((x + 1.) * .5 - mean) / std, y, g, i)

    counter = [resume_steps]

    def step(model, batch, opt, method, device):
        # Exact update-index RNG makes dropout/stochastic depth resumable even
        # across fresh Python processes. It does not reinitialize parameters.
        common.seed_all((c['seed'] + 1000003 * counter[0]) % (2 ** 32))
        counter[0] += 1
        x, y, _, _ = batch; x = x.to(device); y = y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = model.loss(x, y, x.flip(-1) if method == 'core' else None)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite objective')
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        if not any(g is not None for g in grads) or any(g is not None and not torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Missing all gradients or nonfinite gradient')
        opt.step(); return float(loss.detach())

    @torch.inference_mode()
    def predict(model, rows, cfg, device):
        model.eval(); scores = []
        for x, _, _, _ in DataLoader(Images(rows, cfg['size'], True), batch_size=cfg['validation_batch'], num_workers=cfg['workers']):
            scores.extend(model(x.to(device)).cpu().tolist())
        if len(scores) != len(rows) or not np.isfinite(scores).all():
            raise ValueError('Invalid scores')
        return scores

    base.VERSION = VERSION
    base.hashes = hashes
    base.make_model = lambda upstream, weights=None: zoo.Adapter(c['method'], upstream, weights, fixture=fixture)
    base.Images = Images; base.step = step; base.predict = predict
    return base


def smoke(method, sources, device, out):
    out = common.fresh_dir(out); base.setup(17)
    model = zoo.Adapter(method, sources, fixture=True).to(device)
    model.train(); x = torch.randn(2, 3, 224, 224, device=device); y = torch.tensor([0, 1], device=device)
    loss = model.loss(x, y, x.flip(-1) if method == 'core' else None)
    loss.backward()
    gradients = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
    if not torch.isfinite(loss) or not any(g is not None and g.abs().sum() > 0 for _, g in gradients):
        raise RuntimeError('No finite nonzero synthetic gradient')
    if any(g is not None and not torch.isfinite(g).all() for _, g in gradients):
        raise RuntimeError('Nonfinite synthetic gradient')
    model.eval()
    with torch.no_grad():
        both = model(x); singles = torch.cat([model(x[i:i+1]) for i in range(2)])
    torch.testing.assert_close(both, singles, rtol=1e-4, atol=1e-5)
    receipt = dict(status='PASS', method=method, device=str(device), source_hashes=hashes(),
                   parameters=model.counts(), shape=list(both.shape), finite_gradient=True,
                   missing_gradient_parameters=[n for n, g in gradients if g is None],
                   batch_independence_checked=True, synthetic_fixture=True, pretrained_weights_loaded=False,
                   benchmark_images_read=False, performance_evidence=False)
    common.write_json(out / 'SMOKE_RESULT.json', receipt); print(json.dumps(receipt), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['bind', 'preflight', 'smoke', 'dry-run', 'train'])
    p.add_argument('--config'); p.add_argument('--out', required=True); p.add_argument('--sources')
    p.add_argument('--method', choices=zoo.TRAINABLE_METHODS); p.add_argument('--weights')
    p.add_argument('--weights-sha256'); p.add_argument('--preflight')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--receipt'); p.add_argument('--software-receipt'); p.add_argument('--resume', action='store_true')
    p.add_argument('--steps', type=int, default=20); p.add_argument('--budget-hours', type=float)
    a = p.parse_args(); device = torch.device(a.device)
    if device.type=='cuda' and a.command not in ('bind','preflight') and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError('Exactly one bound CUDA device required')
    if a.command == 'smoke':
        if not a.method or not a.sources: p.error('smoke needs --method and --sources')
        smoke(a.method, a.sources, device, a.out); return
    if not a.config: p.error('--config required')
    if a.command == 'bind':
        c = json.loads(Path(a.config).read_text())
        if c['source_hashes'] != hashes(): raise ValueError('Stale source template')
        if not a.weights or not a.weights_sha256 or common.sha(a.weights) != a.weights_sha256:
            raise ValueError('Supply independently verified local weights path and hash')
        c['weights'] = str(Path(a.weights).resolve()); c['weights_sha256'] = a.weights_sha256
        if Path(a.out).exists(): raise ValueError('Preserve existing bound config')
        common.write_json(Path(a.out), c); return
    c = load_config(a.config)
    if a.command == 'preflight':
        configure(c); base.preflight(c, a.out); return
    if not a.sources or not a.preflight: p.error('--sources and --preflight required')
    steps = 0
    if a.resume:
        ck = torch.load(Path(a.out) / 'last.pt', map_location='cpu', weights_only=True)
        steps = ck['steps']; del ck
    configure(c, steps)
    rows, val = base.checked(c, a.preflight)
    if a.command == 'dry-run':
        if a.steps < 3: p.error('At least three profiling steps required')
        base.dry_run(c, rows, a.sources, a.out, device, a.steps); return
    for path, kind in ((a.receipt, 'dry'), (a.software_receipt, 'software')):
        if not path: p.error('Both real-input and software receipts are required')
        receipt = json.loads(Path(path).read_text())
        if receipt.get('status') != 'PASS' or receipt.get('source_hashes') != hashes() or receipt.get('device') != str(device) or receipt.get('method') != c['method']:
            raise ValueError('Missing or stale device/method/source receipt')
        if kind == 'dry' and receipt.get('config_digest') != common.digest(c):
            raise ValueError('Real-input receipt does not match this configuration')
    base.train(c, rows, val, a.sources, a.out, device, resume=a.resume, budget_hours=a.budget_hours)


if __name__ == '__main__':
    main()
