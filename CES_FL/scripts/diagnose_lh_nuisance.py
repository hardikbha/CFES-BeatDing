#!/usr/bin/env python3
"""Zero-update TRAIN diagnostics and synthetic counterexamples for L_H.

No TEST input, target-domain calibration, label inference or optimizer step.
Synthetic mode proves only limitations of the objective, not model failures.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / 'src'
REFERENCE_SHA = {
    'resnet50': '72d368a439658c601a5fe4d717237b1561190e85c13c70f6db3a441bc5cdbe78',
    'xception': '08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177',
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def synthetic(m, out):
    scale = []
    for gain in (1., 10.):
        s, t = torch.tensor([-1.]) * gain, torch.tensor([-.9]) * gain
        response = m.response(s, t)
        scale.append({'logit_gain': gain, 'response': response.item(),
                      'sensitivity_loss': m.sensitivity(response, .5).item()})
    sham = torch.tensor([-2., 2.]); treated = sham + .6
    rr = m.response(sham, treated)
    alignment = m.response_alignment(rr, torch.tensor([0, 1])).item()
    # The two synthetic groups have equally sized procedural responses but
    # all-real original scores fall on opposite sides of threshold zero.
    offset = {'response_alignment_loss': alignment,
              'real_group_positive_rates_at_zero': [0., 1.],
              'interpretation': 'Response alignment alone does not remove additive group score offsets.'}
    torch.manual_seed(17)
    textures = []
    def score(x):
        # A fixed hand-written texture score, no learned forensic semantics.
        roughness = (x[:, 1:, :] - x[:, :-1, :]).abs().mean() + (x[:, :, 1:] - x[:, :, :-1]).abs().mean()
        return -64 * roughness
    for i in range(4):
        x = torch.rand(3, 96, 96) * .6 + .2
        s, t, _, _ = m.intervention(x, 'local_resample', 1., 17+i)
        r = m.response(score(s)[None], score(t)[None])
        noises = {case: float(score(m.nuisance_view(x, case)) - score(x)) for case in m.NUISANCE_BANK}
        textures.append({'texture_index': i, 'procedural_response': float(r),
                         'sensitivity_loss': float(m.sensitivity(r, .5)),
                         'nuisance_only_logit_changes': noises})
    result = {'data': 'Synthetic tensors only; no face images or trained checkpoint',
              'gain_counterexample': scale, 'group_offset_counterexample': offset,
              'texture_only_counterexample': textures,
              'conclusion': 'These demonstrate possible shortcuts, not their actual use by L_H.',
              'trained_model_tested': False, 'scientific_source_sha256': digest(m.__file__)}
    assert scale[0]['sensitivity_loss'] > .39 and scale[1]['sensitivity_loss'] == 0
    assert alignment < 1e-6
    assert all(v['sensitivity_loss'] == 0 for v in textures)
    write(out / 'SYNTHETIC_COUNTEREXAMPLES.json', result)
    print(json.dumps({'status': 'COUNTEREXAMPLES_CONFIRMED', 'trained_model_tested': False}))


def statistics(values):
    values = np.asarray(values, dtype=float)
    return {'n': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
            'p10': float(np.quantile(values, .1)), 'p90': float(np.quantile(values, .9))}


def gradients(ref, model, rows, selected, c, device, arm):
    """One matched TRAIN batch, no optimizer update; weighted component gradients."""
    c = {**c, 'amp': False}
    if 'nuisance_mode' in c: c['nuisance_mode'] = 'none'
    samples = [ref.Originals(rows, c, pairs=True)[i] for i in selected[:32]]
    b = {k: torch.stack([s[k] for s in samples]) if torch.is_tensor(samples[0][k])
         else torch.tensor([s[k] for s in samples]) for k in samples[0]}
    active, _ = ref.group_inventory(rows, c)
    aux = ref.auxiliary(rows, active, c, c['axes'].index('gender_race'), 0)
    reals = sorted([r for r in rows if r['label'] == 0],
                   key=lambda r: ref.seed_for(c['seed'], 'train-calibration', r['image']))[:c['train_calibration_size']]
    threshold = ref.calibrate(ref.predict(model, reals, c, device), c['target_fpr'])
    model.train()
    losses, _ = ref.train_step(model, b, aux, ref.original_weights(rows, c['balance_axis']), c, arm,
                              threshold['threshold_logit'], device)
    k = c['aux_every']
    parts = {
        'BCE': losses['classification'],
        'fairness': k*c['lambda_fair']*losses['error_fairness'],
        'sensitivity': c['lambda_sensitivity']*(losses['original_sensitivity'] + k*losses['aux_sensitivity']),
        'sham_control': c['lambda_sensitivity']*c['sham_weight']*(losses['original_control_stability'] + k*losses['aux_control_stability']),
        'alignment': k*c['lambda_alignment']*losses['response_alignment'],
    }
    params = [p for p in model.parameters() if p.requires_grad]
    collected = {}; info = {}
    for name, loss in parts.items():
        gg = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        collected[name] = tuple(None if g is None else g.detach().float() for g in gg)
        norm2 = sum(float(g.square().sum()) for g in collected[name] if g is not None)
        info[name] = {'weighted_loss': float(loss.detach()), 'gradient_norm': norm2**.5}
    for name, gg in collected.items():
        dot = sum(float((g*b).sum()) for g, b in zip(gg, collected['BCE']) if g is not None and b is not None)
        denom = info[name]['gradient_norm'] * info['BCE']['gradient_norm']
        info[name]['cosine_to_BCE'] = max(-1., min(1., dot/denom)) if denom else None
    model.zero_grad(set_to_none=True); model.eval()
    return {'components': info, 'threshold': threshold, 'optimizer_updates': 0,
            'scope': 'One fixed stratified clean TRAIN main batch and original joint-group auxiliary recipe. Not production sampling frequencies or causal attribution.'}


def checkpoint_run(m, args, out):
    fn = 'ces_fl_head_ablation_resnet50.py' if args.backbone == 'resnet50' else 'ces_fl_head_ablation.py'
    refpath = SRC / fn
    if digest(refpath) != REFERENCE_SHA[args.backbone]: raise ValueError('Reference source changed')
    if args.model_family == 'pilot':
        if args.backbone != 'resnet50': raise ValueError('This pilot has ResNet only')
        refpath = Path(m.__file__)
    if digest(args.checkpoint) != args.checkpoint_sha256: raise ValueError('Checkpoint changed')
    ref = load(refpath, 'pinned_reference')
    model, ck = ref.load_trained(args.checkpoint, torch.device(args.device))
    if ck['arm'] not in ('A', 'D'):
        raise ValueError('Diagnostic requires an A/H checkpoint')
    if ck['arm'] == 'D' and ck['config']['error_control'] != 'hierarchical_real':
        raise ValueError('Diagnostic requires hierarchical H, not legacy D')
    if ck['source_sha256'] != digest(refpath): raise ValueError('Checkpoint source binding differs')
    c = ck['config']
    if digest(c['train']) != c['train_sha256']: raise ValueError('TRAIN manifest changed')
    rows = ref.read_manifest(c['train'])
    fitted = {r['image']: r for r in ck['fit_membership']}
    cells = defaultdict(list)
    for i, r in enumerate(rows):
        saved = fitted.get(r['image'])
        if saved is None or not saved.get('image_sha256'): raise ValueError('Missing recorded TRAIN image hash')
        r['image_sha256'] = saved['image_sha256']
        if r['gender_race'] != 'unknown': cells[(r['gender_race'], r['label'])].append(i)
    if len(cells) != 16: raise ValueError('Expected eight supported TRAIN joint groups and two classes')
    selected = []
    for key, pool in sorted(cells.items()):
        ordered = sorted(pool, key=lambda i: ref.seed_for(17, 'nuisance-diagnostic-v1', rows[i]['image']))
        unique, repeats, tokens = [], [], set()
        for i in ordered:
            token = ref.recorded_contributor(rows[i])
            (repeats if token in tokens else unique).append(i); tokens.add(token)
        chosen = (unique + repeats)[:args.per_cell]
        if len(chosen) != args.per_cell: raise ValueError('Insufficient supported TRAIN cell')
        selected.extend(chosen)
    records = []; model.eval()
    with torch.inference_mode():
        for ix in selected:
            row = rows[ix]; x = ref.load_pixels(row, c['size'])
            xx = [m.nuisance_view(x, nu) for nu in m.NUISANCE_BANK]
            z = model(torch.stack([ref.normalize(v) for v in xx]).to(args.device))['logit'].float().cpu()
            for j, nu in enumerate(m.NUISANCE_BANK):
                records.append({'image': row['image'], 'group': row['gender_race'], 'label': row['label'],
                                'kind': 'original', 'nuisance': nu, 'logit': float(z[j]),
                                'nuisance_delta': float(z[j]-z[0])})
            if row['label'] != 0: continue
            for op in ref.OPERATORS:
                for strength in ref.STRENGTHS:
                    s, t, _, _ = ref.intervention(x, op, strength, ref.seed_for(17, 'diagnostic-pair', row['image'], op, strength))
                    views = [m.common_nuisance(x, s, t, nu) for nu in m.NUISANCE_BANK]
                    logits = model(torch.stack([ref.normalize(v) for triple in views for v in triple]).to(args.device))['logit'].float().cpu().reshape(-1, 3)
                    base_r = float(ref.response(logits[0, 1:2], logits[0, 2:3]))
                    for j, nu in enumerate(m.NUISANCE_BANK):
                        zz, zs, zt = logits[j]; response = float(ref.response(zs[None], zt[None]))
                        records.append({'image': row['image'], 'group': row['gender_race'], 'label': 0,
                            'kind': 'pair', 'operator': op, 'strength': strength, 'nuisance': nu,
                            'response': response, 'response_change': response-base_r,
                            'sensitivity_loss': max(0., c['margin']-response),
                            'raw_sham_gap': float(abs(zs-zz)),
                            'legacy_sham_loss': float(abs(torch.asinh(zs)-torch.asinh(zz))),
                            'pixel_gap': float((views[j][2]-views[j][1]).abs().mean())})
    summaries = []
    grouped = defaultdict(list)
    for r in records:
        key = (r['kind'], r['group'], r['label'], r.get('operator'), r.get('strength'), r['nuisance'])
        grouped[key].append(r)
    for key, rr in grouped.items():
        fields = ('logit', 'nuisance_delta') if key[0] == 'original' else ('response', 'response_change', 'sensitivity_loss', 'raw_sham_gap', 'legacy_sham_loss', 'pixel_gap')
        summaries.append({'kind': key[0], 'group': key[1], 'label': key[2], 'operator': key[3],
                          'strength': key[4], 'nuisance': key[5], **{f: statistics([r[f] for r in rr]) for f in fields}})
    # Interleave real/fake groups in the small gradient batch (not all real first).
    grad_indices = [i for offset in range(args.per_cell) for i in [selected[cell*args.per_cell+offset] for cell in range(16)]]
    before_buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    grad = gradients(ref, model, rows, grad_indices, c, torch.device(args.device), ck['arm'])
    if any(not torch.equal(v, dict(model.named_buffers())[n]) for n, v in before_buffers.items()):
        raise RuntimeError('Diagnostic changed model buffers')
    write(out/'rows_PRIVATE.json', records)
    write(out/'TRAIN_DIAGNOSTICS.json', {'status': 'COMPLETE', 'checkpoint_sha256': args.checkpoint_sha256,
        'reference_source_sha256': digest(refpath), 'model_family': args.model_family, 'arm': ck['arm'], 'diagnostic_source_sha256': digest(__file__),
        'transform_source_sha256': digest(m.__file__), 'train_manifest_sha256': c['train_sha256'],
        'sampled_originals': len(selected), 'per_joint_group_class': args.per_cell, 'summary': summaries,
        'gradient_diagnostic': grad, 'test_read': False, 'optimizer_updates': 0,
        'validity': 'Procedural probes and nuisance candidates only; no proven forensic or demographic disentanglement.'})
    print(json.dumps({'status': 'TRAIN_DIAGNOSTICS_COMPLETE', 'sampled_originals': len(selected), 'test_read': False, 'optimizer_updates': 0}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['synthetic', 'checkpoint'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backbone', choices=['xception', 'resnet50'], default='resnet50')
    p.add_argument('--model-family', choices=['original', 'pilot'], default='original')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--checkpoint-sha256')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--per-cell', type=int, default=8)
    args = p.parse_args(); torch.set_num_threads(2)
    if args.out.exists(): raise ValueError('Preserve earlier diagnostic outputs')
    if args.mode == 'checkpoint' and (not args.checkpoint or not args.checkpoint_sha256 or args.per_cell < 2):
        p.error('Checkpoint, hash and at least two examples per group/class required')
    args.out.mkdir(parents=True)
    m = load(SRC/'ces_fl_nuisance_resnet50.py', 'nuisance_model')
    if args.mode == 'synthetic': synthetic(m, args.out)
    else: checkpoint_run(m, args, args.out)


if __name__ == '__main__': main()
