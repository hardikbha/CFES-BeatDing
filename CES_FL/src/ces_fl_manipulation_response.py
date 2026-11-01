#!/usr/bin/env python3
"""Experimental controlled manipulation response; not a validated causal method.

Uses the adjacent hash-pinned CES-FL architecture. No model downloads, image
generation, automatic approvals, test evaluation or cluster submission.
Edited bases are NOT assigned authenticity labels. Only original TRAIN images
receive BCE labels. Quartets require independently documented operation validity.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import ces_fl_counterfactual_5050 as base

VERSION = 'CESFL_MANIPULATION_RESPONSE_v1'
BASE_SHA = '3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab'
ARMS = ('bce', 'fair', 'nochange', 'score', 'response')
ROLES = ('x0', 'xa', 't0', 'ta')


def hashes():
    if base.sha(base.__file__) != BASE_SHA:
        raise ValueError('Architecture dependency changed')
    return {'trainer': base.sha(__file__), 'architecture': BASE_SHA}


def read_config(path):
    c = json.loads(Path(path).read_text())
    if c['format'] != VERSION:
        raise ValueError('Wrong configuration format')
    for k in ('epochs', 'pair_batch', 'min_per_group_direction'):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError('Invalid ' + k)
    for k in ('lambda_fair', 'lambda_pair', 'margin', 'temperature'):
        if not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError('Invalid ' + k)
    if c['fairness_axis'] != 'gender_race' or c['attribute'] != 'age':
        raise ValueError('This initial protocol is age-only, gender_race fairness')
    if c['pretrain']['pretrain_epochs'] != 50 or c['epochs'] != 50:
        raise ValueError('Production budget is 50+50; amend protocol explicitly')
    if any(c['pretrain'][k] for k in ('pair_bank', 'pretrain_checkpoint')):
        raise ValueError('Pretrain configuration must remain unbound')
    hashes()
    return c


def audit_bank(path, c, rows, val, synthetic=False):
    """Verify recorded bytes/associations; cannot prove the reviewer's assertions."""
    p = json.loads(Path(path).read_text())
    if p['format'] != VERSION or p.get('synthetic_fixture', False) != synthetic:
        raise ValueError('Bank format/fixture role invalid')
    if p['train_sha256'] != c['pretrain']['input_hashes']['train']:
        raise ValueError('Wrong TRAIN bank')
    index = {r['image']: r for r in rows}
    train_ids = {k: {v for r in rows for v in r[k].split(';')} for k in ('identity_ids', 'source_ids')}
    val_ids = {k: {v for r in val for v in r[k].split(';')} for k in train_ids}
    forbidden = {r['image'] for r in rows + val}
    used, ids, sources, quota = set(), set(), set(), Counter()
    for q in p['quartets']:
        if q['id'] in ids:
            raise ValueError('Duplicate quartet ID')
        ids.add(q['id'])
        source = q['source_image']
        if source not in index or index[source]['label'] != 0:
            raise ValueError('Quartet base must be an original real TRAIN image')
        if source in sources:
            raise ValueError('One quartet per unique source in this pilot')
        sources.add(source)
        if base.sha(source) != q['source_sha256']:
            raise ValueError('Source bytes changed')
        if q['attribute'] != c['attribute'] or q['direction'] not in ('older', 'younger'):
            raise ValueError('Wrong demographic intervention')
        group = index[source][c['fairness_axis']]
        if group in ('', 'unknown'):
            raise ValueError('Unknown group')
        quota[(group, q['direction'])] += 1
        review = q['review']
        if not all(base.flag(review.get(k, '')) for k in
                   ('approved', 'identity_preserved', 'attribute_valid', 'operation_valid',
                    'matched_processing', 'unrelated_changes_assessed')):
            raise ValueError('Missing substantive quartet review')
        if not review['reviewer'].strip() or not review['evidence'].strip():
            raise ValueError('Review evidence is required')
        # Reject recorded VAL contributors even if the benchmark allows overlap.
        for k in train_ids:
            values = set(q[k].split(';')) - {''}
            if (not values or not values <= train_ids[k] or values & val_ids[k]
                    or not set(index[source][k].split(';')) <= values):
                raise ValueError('Contributor absent from TRAIN, missing or overlaps VAL')
        with base.Image.open(source) as im:
            size = im.size
        for role in ROLES:
            asset = q[role]
            image = str(Path(asset['image']).resolve())
            if image != asset['image'] or image in forbidden or image in used:
                raise ValueError('Reused/noncanonical/original candidate path')
            used.add(image)
            if base.sha(image) != asset['sha256']:
                raise ValueError('Candidate bytes changed')
            with base.Image.open(image) as im:
                im.load()
                if im.size != size:
                    raise ValueError('Geometry mismatch')
        for role in ('x0', 'xa'):
            if q[role]['parent_sha256'] != q['source_sha256']:
                raise ValueError('Chained/wrong source edit')
        if q['x0']['condition'] != 'nochange' or q['xa']['condition'] != q['direction']:
            raise ValueError('Wrong editor control or direction')
        for k in ('editor_revision', 'editor_settings_sha256', 'seed'):
            if q['x0'][k] != q['xa'][k]:
                raise ValueError('Unmatched editor settings')
        for role, parent in (('t0', 'x0'), ('ta', 'xa')):
            if q[role]['parent_sha256'] != q[parent]['sha256']:
                raise ValueError('Manipulation parent mismatch')
            if not q[role]['operation_receipt'].strip():
                raise ValueError('Missing manipulation receipt')
        for k in ('operator', 'operator_code_sha256', 'settings_sha256', 'donor_sha256', 'seed'):
            if not str(q['t0'][k]).strip() or q['t0'][k] != q['ta'][k]:
                raise ValueError('Unmatched manipulation operation')
        for role in ROLES:
            if q[role]['sha256'] == q[role]['parent_sha256']:
                raise ValueError('Byte-identical no-op is not a valid intervention')
    groups = {r[c['fairness_axis']] for r in rows if r['label'] == 0}
    if not p['quartets'] or 'unknown' in groups or any(
        quota[(g, d)] < c['min_per_group_direction'] for g in groups for d in ('older', 'younger')
    ):
        raise ValueError('Missing prespecified unique-source group/direction coverage')
    return p['quartets'], {'status': 'PASS', 'quartets': len(ids), 'unique_sources': len(sources),
        'real_train_sources': sum(r['label'] == 0 for r in rows),
        'coverage': {f'{g}:{d}': n for (g, d), n in sorted(quota.items())},
        'semantic_validity_independently_proven_by_code': False,
        'recorded_val_contributors_excluded': True, 'synthetic_fixture': synthetic}


def preflight(c, arm, synthetic=False):
    hashes()
    rows, val, weights, splits = base.data_preflight(c['pretrain'], synthetic)
    if any(r[c['fairness_axis']] in ('', 'unknown') for r in rows):
        raise ValueError('Unknown fairness groups')
    bank, coverage = [], None
    if arm in ('nochange', 'score', 'response'):
        if not c['bank'] or base.sha(c['bank']) != c['bank_sha256']:
            raise ValueError('A reviewed hashed quartet bank is required')
        bank, coverage = audit_bank(c['bank'], c, rows, val, synthetic)
    if not c['pretrain_run']:
        raise ValueError('Completed 50-epoch P dependency is not bound')
    p = Path(c['pretrain_run'])
    if not (p/'DONE').is_file() or base.sha(p/'best.pt') != c['pretrain_sha256']:
        raise ValueError('Incomplete/changed pretraining')
    ck = torch.load(p/'best.pt', map_location='cpu', weights_only=True)
    receipt = json.loads((p/'RUN_VERIFIED.json').read_text())
    summary = json.loads((p/'summary.json').read_text())
    if (ck['format'] != base.VERSION or ck['stage'] != 'pretrain' or ck['config'] != c['pretrain']
            or ck['source_sha256'] != BASE_SHA or ck['synthetic_fixture'] != synthetic
            or summary['completed_epochs'] != c['pretrain']['pretrain_epochs']
            or receipt['status'] != 'PASS' or receipt['checkpoint_sha256'] != c['pretrain_sha256']):
        raise ValueError('Pretrain dependency mismatch')
    return rows, val, weights, splits, bank, coverage, ck


def objective(z, y, weights, groups, arm, c, q=None):
    if arm not in ARMS or z.ndim != 1 or z.shape != y.shape or z.shape != weights.shape or z.shape != groups.shape:
        raise ValueError('Invalid objective inputs')
    if (not torch.isfinite(z).all() or not torch.isfinite(weights).all()
            or not ((y == 0) | (y == 1)).all() or (weights <= 0).any()):
        raise ValueError('Invalid scores/labels/weights')
    zero = z.sum()*0
    cls = (weights*F.binary_cross_entropy_with_logits(z, y, reduction='none')).mean()
    real = y == 0
    observed = torch.unique(groups[real])
    fairness = zero
    if arm != 'bce' and len(observed) > 1:
        # Differentiable surrogate at fixed logit=0. TRAIN only, not validation threshold.
        rates = torch.stack([torch.sigmoid(z[real & (groups == g)] / c['temperature']).mean() for g in observed])
        pooled = torch.sigmoid(z[real] / c['temperature']).mean()
        fairness = (rates-pooled).abs().mean()
    pair, alignment, sensitivity = zero, zero, zero
    if arm in ('nochange', 'score', 'response'):
        if q is None or not q['x0'].numel():
            raise ValueError('Missing quartet forward pass')
        if any(v.shape != q['x0'].shape or not torch.isfinite(v).all() for v in q.values()):
            raise ValueError('Malformed quartet logits')
        if arm == 'nochange':
            pair = (torch.asinh(q['original'])-torch.asinh(q['x0'])).abs().mean()
        elif arm == 'score':
            pair = (torch.asinh(q['original'])-torch.asinh(q['xa'])).abs().mean()
        else:
            # Apply asinh to each operation response, not to each raw score:
            # a common additive editor offset cancels within each response.
            m0 = torch.asinh(q['t0']-q['x0'])
            ma = torch.asinh(q['ta']-q['xa'])
            alignment = (m0-ma).abs().mean()
            sensitivity = (F.relu(c['margin']-m0)+F.relu(c['margin']-ma)).mean()/2
            pair = alignment+sensitivity
    return {'total': cls+c['lambda_fair']*fairness+c['lambda_pair']*pair,
        'classification': cls, 'fairness': fairness, 'pair': pair,
        'alignment': alignment, 'sensitivity': sensitivity}, len(observed)


def select_quartets(bank, seed, epoch, step, count):
    # Explicit bounded-pool diagnostic; never claim full original-pair coverage.
    rng = random.Random(f'{seed}:quartets:{epoch}:{step}')
    return rng.sample(bank, min(count, len(bank)))


def quartet_forward(model, bank, arm, reader, device):
    roles = ('original', 'x0') if arm == 'nochange' else ('original', 'x0', 'xa') if arm == 'score' else ROLES
    result = {}
    for role in roles:
        paths = [q['source_image'] if role == 'original' else q[role]['image'] for q in bank]
        result[role] = model(torch.stack([reader.load(p) for p in paths]).to(device))['logit']
    return result


def make_model(c, checkpoint, device):
    p = c['pretrain']
    base.seed_all(p['seed'])
    m = base.DiagramCES('xception', hidden=p['hidden'], gate_hidden=p['gate_hidden'], pretrained=False,
                       encoder_policy='trainable', bn_policy='frozen', head_seed=p['head_seed'])
    m.encoder.load_state_dict({k[8:]: v for k, v in checkpoint['model'].items() if k.startswith('encoder.')}, strict=True)
    return m.to(device)


def run(c, arm, out, device, dry=False, resume=False, synthetic=False):
    rows, val, weights, splits, bank, coverage, pre = preflight(c, arm, synthetic)
    out = Path(out).resolve()
    model = make_model(c, pre, device)
    del pre
    opt = base.build_optimizer(model, c['pretrain'])
    p = c['pretrain']
    reader = base.PairImages([], size=p['size'])
    group_ids = {g: i for i, g in enumerate(sorted({r[c['fairness_axis']] for r in rows}))}
    start, best, uses = 0, -math.inf, Counter()
    if resume:
        if dry or (out/'DONE').exists():
            raise ValueError('Cannot resume dry/completed run')
        ck = torch.load(out/'last.pt', map_location='cpu', weights_only=True)
        if (ck['format'] != VERSION or ck['hashes'] != hashes() or ck['config'] != c or ck['arm'] != arm
                or ck['synthetic_fixture'] != synthetic):
            raise ValueError('Resume provenance mismatch')
        history = [json.loads(l) for l in (out/'history.jsonl').read_text().splitlines()]
        if [r['epoch'] for r in history] != list(range(1, ck['epoch']+1)):
            raise ValueError('Partial checkpoint/history; explicit recovery required')
        selected = torch.load(out/'best.pt', map_location='cpu', weights_only=True)
        if (selected['format'] != VERSION or selected['arm'] != arm or selected['synthetic_fixture'] != synthetic
                or selected['epoch'] != ck['best_epoch'] or selected['hashes'] != hashes() or selected['config'] != c):
            raise ValueError('Selected checkpoint mismatch')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        start, best, best_epoch = ck['epoch'], ck['best_auc'], ck['best_epoch']
        uses.update(ck['quartet_usage'])
        del ck, selected
    else:
        base.fresh(out)
        base.write(out/'config.json', c)
        base.write(out/'provenance.json', {'hashes': hashes(), 'arm': arm, 'coverage': coverage,
            'splits': splits, 'parameters': model.counts(), 'synthetic_fixture': synthetic,
            'all_parameters_trainable': True, 'bn_buffers_frozen': True, 'test_read': False})
    for epoch in range(start, c['epochs']):
        base.seed_all(p['seed']+epoch); model.train()
        loader = DataLoader(base.OwnPairs(rows, weights, p, 'pretrain', epoch),
            batch_sampler=base.EpochBatches(len(rows), p['batch_size'], p['seed'], epoch),
            num_workers=p['workers'], generator=torch.Generator().manual_seed(p['seed']+epoch))
        sums, seen, groups_hist = Counter(), Counter(), Counter()
        for step, b in enumerate(loader):
            ix = b['index'].tolist(); n = len(ix)
            groups = torch.tensor([group_ids[rows[i][c['fairness_axis']]] for i in ix], device=device)
            z = model(b['x'].to(device))['logit']
            chosen = select_quartets(bank, p['seed'], epoch, step, c['pair_batch']) if bank else []
            q = quartet_forward(model, chosen, arm, reader, device) if chosen else None
            losses, ng = objective(z, b['y'].float().to(device), b['weight'].float().to(device), groups, arm, c, q)
            opt.zero_grad(set_to_none=True); losses['total'].backward()
            if not torch.isfinite(losses['total']) or any(v.grad is None or not torch.isfinite(v.grad).all() for v in model.parameters()):
                raise RuntimeError('Missing/nonfinite gradients')
            if dry:
                base.write(out/'DRY_RUN.json', {'status': 'PASS', 'hashes': hashes(), 'arm': arm,
                    'device': device, 'optimizer_steps': 0, 'losses': {k: float(v.detach()) for k, v in losses.items()},
                    'quartets': len(chosen), 'synthetic_fixture': synthetic})
                return
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
            seen.update(ix); uses.update(q['id'] for q in chosen); groups_hist[str(ng)] += 1
            for k, v in losses.items(): sums[k] += float(v.detach())*n
            if (step+1) % 100 == 0:
                print(json.dumps({'arm': arm, 'epoch': epoch+1, 'step': step+1}), flush=True)
        if len(seen) != len(rows) or any(n != 1 for n in seen.values()):
            raise RuntimeError('Original exposure mismatch')
        z = base.predict(model, val, device, p['batch_size'], p['workers'], p['size'])
        cal = base.calibrate(z[np.array([r['label'] == 0 for r in val])], p['target_fpr'])
        report = base.metrics(val, z, cal['threshold_logit'], p['min_support'])
        auc = report['overall']['auc']; improved = auc > best
        if improved: best, best_epoch = auc, epoch+1
        ck = {'format': VERSION, 'hashes': hashes(), 'config': c, 'arm': arm, 'epoch': epoch+1,
              'model': model.state_dict(), 'optimizer': opt.state_dict(), 'best_auc': best,
              'best_epoch': best_epoch, 'calibration': cal, 'validation': report,
              'quartet_usage': dict(uses), 'synthetic_fixture': synthetic}
        if improved: base.atomic_checkpoint(out/'best.pt', ck)
        record = {'epoch': epoch+1, 'validation': report, 'losses': {k: v/len(rows) for k, v in sums.items()},
                  'originals': len(seen), 'real_groups_per_batch': dict(groups_hist), 'quartet_draws_cumulative': sum(uses.values()),
                  'unique_quartets_cumulative': len(uses), 'selection': 'validation_auc_only'}
        with (out/'history.jsonl').open('a') as f: f.write(json.dumps(record, allow_nan=False)+'\n')
        base.atomic_checkpoint(out/'last.pt', ck)
        print(json.dumps({'arm': arm, 'epoch': epoch+1, 'val_auc': auc}), flush=True)
    selected = torch.load(out/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(selected['model'])
    replay = base.predict(model, val, device, p['batch_size'], p['workers'], p['size'])
    actual = base.metrics(val, replay, selected['calibration']['threshold_logit'], p['min_support'])
    if actual != selected['validation']:
        raise RuntimeError('Independent selected-checkpoint validation replay mismatch')
    base.write(out/'RUN_VERIFIED.json', {'status': 'PASS', 'checkpoint_sha256': base.sha(out/'best.pt'),
        'hashes': hashes(), 'completed_epochs': c['epochs'], 'arm': arm, 'synthetic_fixture': synthetic,
        'validation_replayed': True, 'test_read': False, 'fairness_improvement_verified': False})
    (out/'DONE').write_text('Training and validation replay finished; no test evaluation.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('preflight', 'dry-run', 'train'):
        p = sub.add_parser(command); p.add_argument('--config', required=True)
        p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--out', required=True)
        p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
        if command == 'train': p.add_argument('--resume', action='store_true')
    p = sub.add_parser('pretrain'); p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda'); p.add_argument('--resume', action='store_true')
    p = sub.add_parser('bind'); p.add_argument('--config', required=True); p.add_argument('--pretrain-run', required=True)
    p.add_argument('--bank'); p.add_argument('--out', required=True)
    a = parser.parse_args(); c = read_config(a.config)
    if a.command == 'pretrain':
        base.train(c['pretrain'], 'pretrain', a.out, a.device, a.resume)
    elif a.command == 'bind':
        c['pretrain_run'] = str(Path(a.pretrain_run).resolve())
        c['pretrain_sha256'] = base.sha(Path(c['pretrain_run'])/'best.pt')
        c['bank'] = str(Path(a.bank).resolve()) if a.bank else None
        c['bank_sha256'] = base.sha(c['bank']) if c['bank'] else None
        preflight(c, 'response' if c['bank'] else 'bce')
        out = base.fresh(a.out); base.write(out/'bound_config.json', c)
    elif a.command == 'preflight':
        rows, val, _, splits, _, coverage, _ = preflight(c, a.arm)
        base.write(base.fresh(a.out)/'PREFLIGHT.json', {'status': 'PASS', 'hashes': hashes(),
            'train_count': len(rows), 'val_count': len(val), 'splits': splits, 'coverage': coverage})
    else:
        run(c, a.arm, a.out, a.device, a.command == 'dry-run', getattr(a, 'resume', False))


if __name__ == '__main__':
    main()
