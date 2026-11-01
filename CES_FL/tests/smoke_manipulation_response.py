#!/usr/bin/env python3
"""Numerical, provenance and end-to-end software tests; no validity claims."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import ces_fl_manipulation_response as mr


def expect_reject(fn):
    try:
        fn()
    except (ValueError, FileNotFoundError, FileExistsError):
        return
    raise AssertionError('Unsafe input was accepted')


def smoke(device, out):
    b = mr.base
    out = b.fresh(out)
    torch.set_num_threads(4)
    b.seed_all(17)
    checks = []
    def passed(name):
        checks.append(name); print('PASS', name, flush=True)
    c = {'lambda_fair': .1, 'lambda_pair': .1, 'margin': .5, 'temperature': 1.}
    z = torch.tensor([-.5, .7, 1.2, -1.], device=device, requires_grad=True)
    y = torch.tensor([0., 0., 1., 1.], device=device)
    w = torch.tensor([1., 2., .5, 1.], device=device)
    groups = torch.tensor([0, 1, 0, 1], device=device)
    q = {k: torch.tensor(v, device=device, requires_grad=True) for k, v in
         {'x0': [1., 3.], 'xa': [-2., 2.], 't0': [1.1, 3.4], 'ta': [-1.7, 2.2]}.items()}
    loss, n = mr.objective(z, y, w, groups, 'response', c, q)
    zz = z.detach().cpu().numpy().astype(float)
    yy = y.cpu().numpy(); ww = w.cpu().numpy()
    cls = np.mean(ww*(np.maximum(zz, 0)-zz*yy+np.log1p(np.exp(-np.abs(zz)))))
    rates = 1/(1+np.exp(-zz[:2])); fair = np.mean(np.abs(rates-rates.mean()))
    m0, ma = np.arcsinh([.1, .4]), np.arcsinh([.3, .2])
    align = np.mean(abs(m0-ma)); sens = np.mean(np.maximum(0, .5-m0)+np.maximum(0, .5-ma))/2
    assert abs(float(loss['total'].detach())-(cls+.1*fair+.1*(align+sens))) < 1e-6
    loss['total'].backward()
    assert n == 2 and all(t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0 for t in [z, *q.values()])
    passed('independent_numpy_reference_and_gradients_to_all_four_roles')
    shifted = {k: v.detach()+(100 if k in ('x0', 't0') else -40) for k, v in q.items()}
    other, _ = mr.objective(z, y, w, groups, 'response', c, shifted)
    torch.testing.assert_close(loss['pair'], other['pair'], atol=1e-5, rtol=1e-5)
    for empty_y in (torch.ones_like(y), torch.zeros_like(y)):
        l, _ = mr.objective(z, empty_y, w, torch.zeros_like(groups), 'fair', c)
        assert l['fairness'].item() == 0 and torch.isfinite(l['total'])
    plain, _ = mr.objective(z, y, w, groups, 'bce', c)
    assert plain['total'].item() == plain['classification'].item()
    fair_only, _ = mr.objective(z, y, w, groups, 'fair', c)
    fg = torch.autograd.grad(fair_only['fairness'], z)[0]
    assert fg[:2].abs().sum() > 0 and fg[2:].abs().sum() == 0
    q1 = {k: torch.zeros(2, device=device) for k in mr.ROLES}
    collapsed, _ = mr.objective(z, y, w, groups, 'response', c, q1)
    assert collapsed['sensitivity'].item() == c['margin']
    passed('additive_offset_cancellation_no_group_cases_and_collapse_penalty')

    with tempfile.TemporaryDirectory(prefix='response_smoke_') as td:
        root = Path(td).resolve(); rng = np.random.default_rng(17)
        paths = {}
        for split, count in [('train', 8), ('val', 4)]:
            rows = []
            for i in range(count):
                path = root/f'{split}{i}.png'
                b.Image.fromarray(rng.integers(0, 256, (75, 75, 3), dtype=np.uint8)).save(path)
                rows.append({'image': str(path), 'label': i % 2, 'identity_ids': f'{split}{i}',
                    'source_ids': f'{split}{i}', **{k: str((i//2) % 2) for k in b.AXES}})
            paths[split] = root/f'{split}.csv'
            with paths[split].open('w', newline='') as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
        # Random local weights only test the software loader, not pretrained accuracy.
        init = b.DiagramCES(pretrained=False)
        weights_path = root/'random_weights.pt'; torch.save(init.encoder.state_dict(), weights_path)
        del init
        p = b.config_default()
        p.update(train=str(paths['train']), val=str(paths['val']), protocol_overlap_file=None,
                 weights=str(weights_path), hidden=8, gate_hidden=4, size=75, workers=0,
                 pretrain_epochs=1, downstream_epochs=1, batch_size=4, min_support=1)
        p['input_hashes'] = {k: b.sha(p[k]) for k in ('train', 'val', 'weights')}
        c.update(format=mr.VERSION, pretrain=p, fairness_axis='gender_race', attribute='age',
                 pair_batch=2, min_per_group_direction=1, epochs=1, bank=None, bank_sha256=None,
                 pretrain_run=None, pretrain_sha256=None)
        train_rows, val_rows = b.read_rows(p['train']), b.read_rows(p['val'])
        bank = {'format': mr.VERSION, 'synthetic_fixture': True, 'train_sha256': p['input_hashes']['train'], 'quartets': []}
        for j, row in enumerate(r for r in train_rows if r['label'] == 0):
            source = row['image']; direction = 'older' if j < 2 else 'younger'
            q = {'id': f'q{j}', 'source_image': source, 'source_sha256': b.sha(source), 'attribute': 'age',
                 'direction': direction, 'identity_ids': row['identity_ids'], 'source_ids': row['source_ids'],
                 'review': {k: True for k in ('approved', 'identity_preserved', 'attribute_valid',
                           'operation_valid', 'matched_processing', 'unrelated_changes_assessed')}}
            q['review'].update(reviewer='SYNTHETIC_FIXTURE', evidence='Not a real approval')
            for role in mr.ROLES:
                path = root/f'q{j}_{role}.png'
                b.Image.fromarray(rng.integers(0, 256, (75, 75, 3), dtype=np.uint8)).save(path)
                q[role] = {'image': str(path), 'sha256': b.sha(path), 'seed': 17}
            for role in ('x0', 'xa'):
                q[role].update(parent_sha256=q['source_sha256'], editor_revision='fixture',
                               editor_settings_sha256='fixture', condition='nochange' if role == 'x0' else direction)
            for role, parent in [('t0', 'x0'), ('ta', 'xa')]:
                q[role].update(parent_sha256=q[parent]['sha256'], operator='synthetic_fixture',
                    operator_code_sha256='fixture', settings_sha256='fixture', donor_sha256='none', operation_receipt='fixture')
            bank['quartets'].append(q)
        bp = root/'bank.json'; b.write(bp, bank)
        _, report = mr.audit_bank(bp, c, train_rows, val_rows, True)
        assert report['quartets'] == 4
        expect_reject(lambda: mr.audit_bank(bp, c, train_rows, val_rows))
        for mutation in ('review', 'parent', 'operation', 'val_contributor', 'source', 'coverage', 'image'):
            bad = deepcopy(bank)
            if mutation == 'review': bad['quartets'][0]['review']['approved'] = False
            if mutation == 'parent': bad['quartets'][0]['ta']['parent_sha256'] = bad['quartets'][0]['source_sha256']
            if mutation == 'operation': bad['quartets'][0]['ta']['settings_sha256'] = 'different'
            if mutation == 'val_contributor': bad['quartets'][0]['identity_ids'] += ';val0'
            if mutation == 'source': bad['quartets'][0]['source_image'] = train_rows[1]['image']
            if mutation == 'coverage': bad['quartets'].pop()
            if mutation == 'image': bad['quartets'][0]['xa']['sha256'] = 'changed'
            path = root/'bad.json'; b.write(path, bad)
            expect_reject(lambda: mr.audit_bank(path, c, train_rows, val_rows, True))
        passed('bank_bytes_roles_training_contributors_review_and_coverage_guards')
        c['bank'], c['bank_sha256'] = str(bp), b.sha(bp)
        b.train(p, 'pretrain', root/'P', device, synthetic=True)
        c['pretrain_run'], c['pretrain_sha256'] = str(root/'P'), b.sha(root/'P/best.pt')
        ck = torch.load(root/'P/best.pt', weights_only=True, map_location='cpu')
        one, two = mr.make_model(c, ck, device), mr.make_model(c, ck, device)
        assert all(torch.equal(v, two.state_dict()[k]) for k, v in one.state_dict().items())
        one.train(); before = {k: v.clone() for k, v in one.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
        one(torch.randn(2, 3, 75, 75, device=device))['logit'].sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in one.parameters())
        assert all(torch.equal(v, one.state_dict()[k]) for k, v in before.items())
        del ck, one, two, before
        passed('pretraining_transfer_identical_initialization_encoder_gradients_frozen_bn')
        for arm in mr.ARMS:
            mr.run(c, arm, root/f'dry_{arm}', device, dry=True, synthetic=True)
            mr.run(c, arm, root/arm, device, synthetic=True)
            receipt = json.loads((root/arm/'RUN_VERIFIED.json').read_text())
            assert receipt['status'] == 'PASS' and receipt['test_read'] is False
            expect_reject(lambda: mr.run(c, arm, root/arm, device, synthetic=True))
        passed('all_five_arms_real_architecture_backward_update_save_reload_validation_replay')
        c['epochs'] = 2
        mr.run(c, 'response', root/'reference', device, synthetic=True)
        original_save = b.atomic_checkpoint
        def interrupt(path, obj):
            original_save(path, obj)
            if Path(path).name == 'last.pt' and obj['epoch'] == 1:
                raise RuntimeError('SIMULATED_INTERRUPT')
        try:
            with patch.object(b, 'atomic_checkpoint', interrupt):
                mr.run(c, 'response', root/'interrupted', device, synthetic=True)
        except RuntimeError as exc:
            assert str(exc) == 'SIMULATED_INTERRUPT'
        mr.run(c, 'response', root/'interrupted', device, resume=True, synthetic=True)
        a = torch.load(root/'reference/last.pt', weights_only=True, map_location='cpu')
        d = torch.load(root/'interrupted/last.pt', weights_only=True, map_location='cpu')
        assert all(torch.equal(v, d['model'][k]) for k, v in a['model'].items())
        assert a['quartet_usage'] == d['quartet_usage'] and a['calibration'] == d['calibration']
        assert b.calibrate([.7]*4, .25)['achieved_fp'] == 0
        passed('epoch_resume_bitwise_equivalence_pair_usage_and_tied_threshold')
    b.write(out/'RESULT.json', {'status': 'PASS', 'device': device, 'checks': checks,
        'hashes': mr.hashes(), 'test_sha256': b.sha(__file__), 'torch': torch.__version__,
        'synthetic_fixture_only': True, 'actual_manipulation_pipeline_tested': False,
        'semantic_validity_verified': False, 'fairness_improvement_verified': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--out', required=True); a = p.parse_args(); smoke(a.device, a.out)
