#!/usr/bin/env python3
"""Bind H0, diagnose TRAIN gradients, and run one gated projection candidate.

No scheduler submission, downloads, approval fabrication or TEST/CAL interface.
The native scientific source is immutable; optimizer handling is bound separately.
"""
import argparse
import copy
from collections import Counter
from contextlib import contextmanager
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / 'proposals/lh_gradient_conflict_20261110_r1'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


f = module('conflict_existing_diagnostics', ROOT / 'scripts/lh_alignment_followup.py')
m = f.m
gp = module('conflict_projection', ROOT / 'src/alignment_gradient_projection.py')


def read(path):
    return json.loads(Path(path).read_text())


def fresh_json(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError('Preserve existing artifact: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    m.write_json(path, value)


def fingerprints():
    frozen = read(REQUEST / 'FROZEN_FILES.json')
    actual = {name: m.sha(ROOT / name) for name in frozen}
    if actual != frozen:
        raise ValueError('Frozen implementation/protocol changed; new proposal required')
    return actual


def plan():
    fingerprints()
    p = read(REQUEST / 'protocol.json')
    if m.sha(f.SOURCE) != p['scientific_source_sha256']:
        raise ValueError('Native scientific source differs')
    if m.sha(REQUEST / 'native_config.json') != p['native_config_sha256']:
        raise ValueError('Native config must remain byte-identical to H0')
    return p


def checkout_guard():
    expected = os.environ.get('CES_EXPECTED_COMMIT', '')
    if len(expected) != 40:
        raise ValueError('Set CES_EXPECTED_COMMIT to the exact pushed commit')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != expected or git('status', '--porcelain'):
        raise ValueError('Use a clean checkout of the exact handoff commit')
    result = subprocess.run(['git', '-C', str(ROOT), 'symbolic-ref', '-q', 'HEAD'], capture_output=True)
    if result.returncode != 1:
        raise ValueError('Production work requires a detached source checkout')
    return expected


def allocation_guard(device):
    if device.type != 'cuda':
        return None
    receipt = os.environ.get('CES_GPU_BINDING_RECEIPT')
    if not receipt or not Path(receipt).is_file():
        raise ValueError('A verified allocation/binding receipt is required')
    a = read(receipt)
    if a.get('synthetic_fixture'):
        raise ValueError('Synthetic allocation receipts cannot authorize GPU work')
    required = {'kind', 'allocation_id', 'gpu_uuid', 'deadline_epoch', 'ledger_scope',
                'cohort_id', 'cohort_started_epoch',
                'ledger_path', 'ledger_sha256', 'verified_remaining_gpu_hours', 'operator_verified'}
    if not required.issubset(a):
        raise ValueError('Incomplete allocation/budget receipt')
    if a['kind'] not in ('PBS', 'AUTHORIZED_POOL_V2') or a['operator_verified'] is not True or not a['allocation_id']:
        raise ValueError('Verify actual scheduler allocation or explicitly authorized pool lease')
    if a['ledger_scope'] != 'ORIGINAL_LH_STUDY' or m.sha(a['ledger_path']) != a['ledger_sha256']:
        raise ValueError('Wrong/stale ledger; FF2 hours cannot be transferred')
    if not str(a['gpu_uuid']).startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES') != a['gpu_uuid']:
        raise ValueError('Bind CUDA_VISIBLE_DEVICES to exactly the verified physical UUID')
    if not math.isfinite(a['verified_remaining_gpu_hours']) or a['verified_remaining_gpu_hours'] <= 0:
        raise ValueError('No verified remaining budget')
    if a['cohort_id'] != 'lh_gradient_conflict_20261110_r1':
        raise ValueError('Wrong cohort allocation')
    if not 0 <= time.time() - float(a['cohort_started_epoch']) or not 0 < float(a['deadline_epoch']) - float(a['cohort_started_epoch']) <= 7200:
        raise ValueError('Use one fixed at-most-two-hour cohort window; do not reset it per stage')
    seconds = float(a['deadline_epoch']) - time.time()
    if not 0 < seconds <= min(2., a['verified_remaining_gpu_hours']) * 3600:
        raise ValueError('Allocation deadline exceeds bounded remaining allowance')
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one visible CUDA device required')
    # This validates a supplied receipt and binding. It does not authenticate
    # scheduler ownership; the operator/controller must verify qstat/lease state.
    return {'receipt_path': str(Path(receipt).resolve()), 'receipt_sha256': m.sha(receipt),
            'gpu_uuid': a['gpu_uuid'], 'allocation_id': a['allocation_id'],
            'cohort_id': a['cohort_id'], 'cohort_started_epoch': a['cohort_started_epoch'],
            'deadline_epoch': a['deadline_epoch'], 'ownership_verified_by_operator': True}


def verify_review(c, rows, path, expected_count):
    m.check_qualification(c, path)
    q = read(path)
    preview, review = read(q['preview_file']), read(q['review_file'])
    if preview['synthetic_fixture'] or preview['source_sha256'] != m.sha(f.SOURCE) or preview['config_digest'] != m.digest(c):
        raise ValueError('Qualification source/config/fixture mismatch')
    all_cases = f.canonical_cases(c, rows)
    expected = {k: v for k, v in all_cases.items() if v['settings']['common_post_nuisance'] == 'identity'}
    if c['nuisance_mode'] != 'none' or len(expected) != expected_count:
        raise ValueError('Expected unchanged H0 88-case scope')
    f.verify_case_inventory(preview, review, expected)
    if q['case_count'] != len(expected) or review['preview_sha256'] != m.sha(q['preview_file']):
        raise ValueError('Qualification count/hash differs')
    for case in preview['cases']:
        if m.sha(case['image']) != case['image_sha256'] or m.sha(case['sheet']) != case['sheet_sha256']:
            raise ValueError('Reviewed source/sheet bytes changed')
    replayed = verify_reviewed_pixels(c, rows, preview)
    return {'qualification_sha256': m.sha(path), 'preview_sha256': m.sha(q['preview_file']),
            'review_sha256': m.sha(q['review_file']), 'cases': len(expected),
            'reviewed_panels_replayed_exactly': replayed,
            'new_approvals_created': False, 'human_review_claimed': False,
            'scope': 'Existing bounded procedural review; not real-forgery validity'}


def verify_reviewed_pixels(c, rows, preview):
    """Recompute tensors and compare existing panel pixels; write no images.

    This detects an environment/preprocessing drift even when the original
    source/sheet file hashes are intact. It grants no new validity approval.
    """
    by_path = {r['image']: r for r in rows}; originals = {}; count = 0; size = c['size']
    for case in preview['cases']:
        if case['image'] not in originals:
            originals[case['image']] = m.load_pixels(by_path[case['image']], size)
        x = originals[case['image']]; settings = case['settings']
        s, t, _, _ = m.intervention(x, settings['operator'], settings['strength'], settings['seed'])
        nx, ns, nt = m.common_nuisance(x, s, t, settings['common_post_nuisance'])
        with m.Image.open(case['sheet']) as im:
            pixels = np.asarray(im.convert('RGB'))
        if pixels.shape != (size + 24, size * 4, 3):
            raise ValueError('Reviewed contact-sheet geometry changed')
        for index, tensor in enumerate((nx, ns, nt, (nt - ns).abs() * 4)):
            expected = m.image_uint8(tensor)
            observed = pixels[24:, index * size:(index + 1) * size]
            if not np.array_equal(expected, observed):
                raise ValueError('Current operator/environment pixels differ from existing review; stop')
            count += 1
    return count


def reference_spec(p):
    return {'scientific_source_sha256': p['scientific_source_sha256'],
            'config_sha256': {'H0': p['native_config_sha256']},
            'checkpoint_sha256': {'H0': p['reference_checkpoint_sha256']}}


def bind(refs_path, out):
    commit = checkout_guard()
    p = plan()
    refs = read(refs_path)
    if set(refs) != {'H0_run', 'H0_preflight', 'H0_qualification'}:
        raise ValueError('Use the exact references template')
    refs = {k: str(Path(v).resolve()) for k, v in refs.items()}
    c = m.load_config(REQUEST / 'native_config.json')
    reference = f.verify_reference(refs['H0_run'], 'H0', reference_spec(p))
    rows, _, _ = m.checked_inputs(c, refs['H0_preflight'])
    qualification = verify_review(c, rows, refs['H0_qualification'], p['reference_cases'])
    fresh_json(out, {'status': 'BOUND', 'commit': commit, 'fingerprints': fingerprints(),
                    'refs': refs, 'reference': reference, 'qualification': qualification,
                    'preflight_sha256': m.sha(Path(refs['H0_preflight']) / 'PREFLIGHT.json'),
                    'test_cal_read': False})
    print('BOUND_UNCHANGED_H0_CONTROL_AND_88_CASE_SCOPE')


def load_binding(path):
    commit = checkout_guard()
    p, b = plan(), read(path)
    if b['status'] != 'BOUND' or b['fingerprints'] != fingerprints() or b['commit'] != commit:
        raise ValueError('Binding belongs to different source/protocol')
    c = m.load_config(REQUEST / 'native_config.json')
    ref = f.verify_reference(b['refs']['H0_run'], 'H0', reference_spec(p))
    if ref != b['reference']:
        raise ValueError('Historical control artifacts changed')
    if b['preflight_sha256'] != m.sha(Path(b['refs']['H0_preflight']) / 'PREFLIGHT.json'):
        raise ValueError('Preflight changed')
    rows, val, pre = m.checked_inputs(c, b['refs']['H0_preflight'])
    if verify_review(c, rows, b['refs']['H0_qualification'], p['reference_cases']) != b['qualification']:
        raise ValueError('Existing review changed')
    return p, b, c, rows, val, pre


def assess(records, settings):
    valid = [r['gradients']['alignment']['all'] for r in records
             if r['gradients']['alignment']['all']['cosine_to_BCE'] is not None
             and r['gradients']['alignment']['all']['norm_ratio_to_BCE'] is not None
             and math.isfinite(r['gradients']['alignment']['all']['cosine_to_BCE'])
             and math.isfinite(r['gradients']['alignment']['all']['norm_ratio_to_BCE'])]
    fraction = sum(r['cosine_to_BCE'] < 0 for r in valid) / len(valid) if valid else None
    ratio = float(np.median([r['norm_ratio_to_BCE'] for r in valid])) if valid else None
    run = bool(len(valid) >= settings['minimum_valid_batches']
               and fraction >= settings['minimum_negative_fraction']
               and ratio >= settings['minimum_median_alignment_to_bce_ratio'])
    return {'run_candidate': run, 'valid_batches': len(valid), 'negative_fraction': fraction,
            'median_alignment_to_bce_ratio': ratio,
            'interpretation': 'Engineering triage, not proof of harmful conflict or significance'}


def diagnose(binding, out, device):
    p, b, c, rows, _, _ = load_binding(binding)
    allocation = allocation_guard(device)
    if device.type == 'cuda' and str(torch.__version__) != b['reference']['torch']:
        raise ValueError('Use the original H0 PyTorch/CUDA environment')
    out = m.fresh_dir(out)
    m.seed_all(17)
    model, ck = m.load_trained(Path(b['refs']['H0_run']) / 'best.pt', device)
    if ck['arm'] != 'D':
        raise ValueError('H0 must be full native D')
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    reals = sorted((r for r in rows if r['label'] == 0),
                   key=lambda r: m.seed_for(c['seed'], 'train-calibration', r['image']))[:2048]
    threshold = m.calibrate(m.predict(model, reals, c, device), c['target_fpr'])
    active, _ = m.group_inventory(rows, c)
    weights, records = m.original_weights(rows, c['balance_axis']), []
    for event in p['diagnostic']['events']:
        if allocation and time.time() >= allocation['deadline_epoch']:
            raise TimeoutError('Diagnostic deadline; no candidate launch')
        batch, aux, indices = f.batch_views(rows, active, c, event, 'identity')
        r = {'event': event, 'axis': aux['axis'], 'operator': aux['operator'], 'strength': aux['strength'],
             'main_plan_digest': m.digest([rows[i]['image'] for i in indices]),
             'main_class_counts': dict(Counter(str(rows[i]['label']) for i in indices)),
             **f.profile_batch(model, batch, aux, weights, c, threshold['threshold_logit'], device)}
        records.append(r)
        print(json.dumps({'event': event, 'axis': aux['axis'],
                          'cosine': r['gradients']['alignment']['all']['cosine_to_BCE']}), flush=True)
    if any(not torch.equal(v, model.state_dict()[k].detach().cpu()) for k, v in before.items()):
        raise RuntimeError('Zero-update diagnostic changed model state')
    result = {'status': 'COMPLETE', 'binding_sha256': m.sha(binding), 'fingerprints': fingerprints(),
              'train_threshold': threshold, 'records': records, 'decision': assess(records, p['diagnostic']),
              'runtime_torch': str(torch.__version__), 'device': str(device), 'allocation': allocation,
              'optimizer_updates': 0, 'test_cal_read': False, 'state_unchanged': True}
    m.write_json(out / 'DIAGNOSTIC.json', result)
    print(json.dumps(result['decision']))


def gate_diagnostic(binding, diagnostic, p, rows):
    d = read(diagnostic)
    if d['status'] != 'COMPLETE' or d['binding_sha256'] != m.sha(binding) or d['fingerprints'] != fingerprints():
        raise ValueError('Stale diagnostic')
    if d.get('optimizer_updates') != 0 or d.get('test_cal_read') is not False or d.get('state_unchanged') is not True:
        raise ValueError('Diagnostic scope/state changed')
    events = [r['event'] for r in d['records']]
    if len(events) != 20 or sorted(events) != p['diagnostic']['events']:
        raise ValueError('Incomplete/duplicate diagnostic events')
    conditions = [(op, strength) for op in m.OPERATORS for strength in m.STRENGTHS]
    for r in d['records']:
        op, strength = conditions[r['event'] // 5]
        if r['axis'] != ['gender', 'race', 'age_bin', 'skin_bin', 'gender_race'][r['event'] % 5] or r['operator'] != op or r['strength'] != strength or r['optimizer_updates'] != 0:
            raise ValueError('Diagnostic plan/coverage differs')
        indices = f.main_indices(rows, r['event'])
        if r['main_plan_digest'] != m.digest([rows[i]['image'] for i in indices]):
            raise ValueError('Diagnostic main sampling differs from native uniform plan')
        if r['main_class_counts'] != dict(Counter(str(rows[i]['label']) for i in indices)):
            raise ValueError('Diagnostic class exposure differs')
    decision = assess(d['records'], p['diagnostic'])
    if d['decision'] != decision or not decision['run_candidate']:
        raise ValueError('Conflict trigger not supported; stop without training')
    return d


def make_fixture_batch(device):
    m.seed_all(17)
    x = torch.rand(4, 3, 224, 224)
    batch = {'x': torch.stack([m.normalize(a) for a in x]), 'y': torch.tensor([0., 1., 0., 1.]),
             's': torch.stack([m.normalize(a * .98) for a in x]),
             't': torch.stack([m.normalize((a + .03).clamp(0, 1)) for a in x]), 'index': torch.arange(4)}
    aux = {'x': batch['x'], 'y': batch['y'], 's': batch['s'][[0, 2]],
           't': torch.stack([m.normalize((x[0] + .04).clamp(0, 1)), m.normalize((x[2] + .08).clamp(0, 1))]),
           'group': torch.tensor([0, 0, 1, 1]), 'priors': torch.ones(2, 2) / 2,
           'axis': 'gender_race', 'hierarchy': {'parent_codes': torch.tensor([0, 1]), 'within_parent_real_priors': torch.eye(2)}}
    return batch, aux


def software_check(out, device):
    """Full-width synthetic backward only. No external weights or private data."""
    if device.type == 'cuda':
        checkout_guard()
    allocation = allocation_guard(device)
    c = {**m.defaults(), 'amp': device.type == 'cuda', 'error_control': 'hierarchical_real', 'nuisance_mode': 'none'}
    batch, aux = make_fixture_batch(device)
    model = m.Detector(policy='exit', head_type='linear').to(device)
    counts = model.parameter_report()
    baseline, reference = copy.deepcopy(model), copy.deepcopy(model)
    for current, native in ((baseline, True), (reference, False)):
        loss, _ = m.train_step(current, batch, aux, torch.ones(4), c, 'D', 0., device)
        opt = m.build_optimizer(current, c)
        if native:
            m.finite_step(loss, current, opt)
        else:
            gp.ProjectedStep(c, mode='native')(loss, current, opt)
    if any(not torch.equal(v, reference.state_dict()[k]) for k, v in baseline.state_dict().items()):
        raise RuntimeError('Projection-disabled update does not match native step exactly')
    del baseline, reference, opt, loss
    before = {n: v.detach().cpu().clone() for n, v in model.named_buffers()}
    losses, _ = m.train_step(model, batch, aux, torch.ones(4), c, 'D', 0., device)
    if not losses['response_alignment'].requires_grad:
        raise RuntimeError('Alignment is detached')
    optimizer = m.build_optimizer(model, c)
    hook = gp.ProjectedStep(c)
    stats = hook(losses, model, optimizer)
    if any(not torch.equal(v, dict(model.named_buffers())[n].detach().cpu()) for n, v in before.items()):
        raise RuntimeError('BN buffers changed')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError('Frozen parameters received gradients')
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
        raise RuntimeError('Missing or nonfinite trainable gradient')
    fresh_json(out, {'status': 'PASS', 'fingerprints': fingerprints(), 'device': str(device),
                     'torch': str(torch.__version__), 'parameters': counts, 'projection': stats,
                     'bn_buffers_unchanged': True, 'data': 'Synthetic 224-pixel software fixture only',
                     'projection_disabled_update_matches_native_exactly': True,
                     'allocation': allocation, 'private_inputs_read': False})
    print('FULL_WIDTH_PROJECTION_BACKWARD_PASS')


def profile(binding, diagnostic, software, out, device):
    p, b, c, rows, val, _ = load_binding(binding)
    gate_diagnostic(binding, diagnostic, p, rows)
    allocation = allocation_guard(device)
    check_software(software, device, b)
    out = m.fresh_dir(out)
    model = m.load_initial(c).to(device)
    optimizer = m.build_optimizer(model, c)
    active, _ = m.group_inventory(rows, c)
    weights = m.original_weights(rows, c['balance_axis'])
    reals = sorted((r for r in rows if r['label'] == 0), key=lambda r: m.seed_for(17, 'train-calibration', r['image']))[:2048]
    threshold = m.calibrate(m.predict(model, reals, c, device), c['target_fpr'])['threshold_logit']
    model.train()
    loader = iter(m.make_loader(rows, c, 0, pairs=True, shuffle=True))
    elapsed = []
    with gp.installed(m, c, deadline_epoch=allocation['deadline_epoch'] if allocation else None) as hook:
        for step in range(60):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            batch = next(loader)
            aux = m.auxiliary(rows, active, c, step // 4, 0) if step % 4 == 0 else None
            losses, _ = m.train_step(model, batch, aux, weights, c, 'D', threshold, device)
            m.finite_step(losses, model, optimizer)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
        summary = hook.summary()
    if len(elapsed) != 60:
        raise ValueError('Insufficient profile batches')
    start = time.perf_counter()
    m.predict(model, val, c, device)  # timing only; do not publish/select profile scores
    if device.type == 'cuda':
        torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - start
    steps = math.ceil(len(rows) / c['batch_size'])
    estimate = 1.5 * (3 * steps * float(np.mean(elapsed[20:])) + 4 * validation_seconds + 120.) / 3600.
    cap = p['budget']['native_training_cap_hours']
    record = {'status': 'PASS' if estimate <= cap else 'BLOCKED_PROFILE_EXCEEDS_CAP',
              'binding_sha256': m.sha(binding), 'diagnostic_sha256': m.sha(diagnostic),
              'software_sha256': m.sha(software), 'fingerprints': fingerprints(),
              'torch': str(torch.__version__), 'device': str(device), 'allocation': allocation,
              'discarded_optimizer_steps': 60, 'warmup_steps': 20, 'measured_steps': 40,
              'cold_plus_warmup_seconds': sum(elapsed[:20]), 'steady_step_seconds': float(np.mean(elapsed[20:])),
              'validation_seconds': validation_seconds, 'conservative_training_replay_hours': estimate,
              'native_training_cap_hours': cap, 'projection': summary,
              'private_scores_saved': False, 'profile_model_discarded': True,
              'note': 'Timing estimate includes actual image/augmentation work inside steps, plus margin; external ledger must include all loading/idle/diagnostic costs.'}
    m.write_json(out / 'PROFILE.json', record)
    print(json.dumps({'status': record['status'], 'estimated_hours': estimate}))


def check_software(path, device, binding):
    r = read(path)
    if r['status'] != 'PASS' or r['fingerprints'] != fingerprints() or r['device'] != str(device):
        raise ValueError('Fresh matching-device software receipt required')
    if r['torch'] != str(torch.__version__) or (device.type == 'cuda' and r['torch'] != binding['reference']['torch']):
        raise ValueError('Use the original H0 runtime and matching software checks')


@contextmanager
def checkpoint_metadata(metadata):
    native = m.atomic_save
    def save(path, value):
        if isinstance(value, dict) and value.get('format') == m.VERSION and 'model' in value:
            value = {**value, 'gradient_update_rule': metadata}
        return native(path, value)
    m.atomic_save = save
    try:
        yield
    finally:
        m.atomic_save = native


def train_candidate(binding, diagnostic, software, profile_path, out, budget_hours):
    device = torch.device('cuda')
    p, b, c, rows, _, _ = load_binding(binding)
    gate_diagnostic(binding, diagnostic, p, rows)
    allocation = allocation_guard(device)
    check_software(software, device, b)
    prof = read(profile_path)
    if prof['status'] != 'PASS' or prof['fingerprints'] != fingerprints() or prof['binding_sha256'] != m.sha(binding) or prof['diagnostic_sha256'] != m.sha(diagnostic) or prof['software_sha256'] != m.sha(software):
        raise ValueError('Profile is stale, blocked or belongs to other checks')
    if prof['torch'] != str(torch.__version__) or prof['device'] != 'cuda':
        raise ValueError('Profile runtime differs')
    for field in ('allocation_id', 'gpu_uuid', 'cohort_id', 'cohort_started_epoch', 'deadline_epoch'):
        if prof['allocation'][field] != allocation[field]:
            raise ValueError('Keep profile and candidate inside the same fixed cohort allocation window')
    if not math.isfinite(budget_hours) or not prof['conservative_training_replay_hours'] <= budget_hours <= p['budget']['native_training_cap_hours']:
        raise ValueError('Full candidate/replay does not fit the native allowance')
    if time.time() + budget_hours * 3600 > allocation['deadline_epoch']:
        raise ValueError('Candidate does not fit actual cohort deadline')
    out = Path(out)
    if out.exists():
        raise ValueError('No automatic rerun/resume; preserve prior candidate')
    metadata = {'format': gp.VERSION, 'mode': 'projected', 'fingerprints': fingerprints(),
                'binding_sha256': m.sha(binding), 'diagnostic_sha256': m.sha(diagnostic),
                'native_arm': 'D', 'same_initialization_as_control': True}
    launch = {**metadata, 'allocation': allocation, 'profile_sha256': m.sha(profile_path),
              'profile_path': str(Path(profile_path).resolve()), 'diagnostic_path': str(Path(diagnostic).resolve()),
              'software_path': str(Path(software).resolve()), 'software_sha256': m.sha(software),
              'budget_hours': budget_hours, 'test_cal_read': False}
    fresh_json(out.parent / (out.name + '_LAUNCH.json'), launch)
    with gp.installed(m, c, deadline_epoch=allocation['deadline_epoch']) as hook, checkpoint_metadata(metadata):
        try:
            m.train(c, b['refs']['H0_preflight'], out, 'D', device,
                    qualification=b['refs']['H0_qualification'], budget_hours=budget_hours)
        finally:
            fresh_json(out.parent / (out.name + '_UPDATE_SUMMARY.json'), hook.summary())


def report(binding, candidate, out):
    p, b, c, rows, _, _ = load_binding(binding)
    root = Path(candidate)
    launch = read(root.parent / (root.name + '_LAUNCH.json'))
    if launch['fingerprints'] != fingerprints() or launch['binding_sha256'] != m.sha(binding):
        raise ValueError('Candidate launch provenance differs')
    if launch['diagnostic_sha256'] != m.sha(launch['diagnostic_path']):
        raise ValueError('Launch diagnostic changed')
    gate_diagnostic(binding, launch['diagnostic_path'], p, rows)
    if not (root / 'DONE').is_file():
        raise ValueError('Incomplete candidate; do not compare as a full run')
    verified, history, scores = [read(root / name) for name in ('RUN_VERIFIED.json', 'history.json', 'validation_scores_PRIVATE.json')]
    if verified['status'] != 'COMPLETED_VERIFIED' or verified['arm'] != 'D' or verified.get('synthetic_fixture') or verified.get('test_evaluated'):
        raise ValueError('Wrong candidate completion')
    if verified['source_sha256'] != m.sha(f.SOURCE) or verified['config_digest'] != m.digest(c) or m.sha(root / 'config.json') != p['native_config_sha256']:
        raise ValueError('Candidate native source/config changed')
    if [r['epoch'] for r in history] != [1, 2, 3] or verified['epochs_completed'] != 3:
        raise ValueError('Candidate epoch budget changed')
    winner = max(history, key=lambda r: r['validation']['overall']['auc'])
    if winner['epoch'] != verified['selected_epoch'] or winner['validation'] != verified['validation']:
        raise ValueError('Checkpoint selection changed')
    checksum = m.sha(root / 'best.pt')
    if checksum != verified['selected_checkpoint_sha256'] or scores['checkpoint_sha256'] != checksum:
        raise ValueError('Candidate checkpoint hash differs')
    ck = torch.load(root / 'best.pt', map_location='cpu', weights_only=True)
    if ck.get('gradient_update_rule') != {k: launch[k] for k in ('format', 'mode', 'fingerprints', 'binding_sha256', 'diagnostic_sha256', 'native_arm', 'same_initialization_as_control')}:
        raise ValueError('Checkpoint lacks the new optimizer-rule provenance')
    del ck
    if scores['split'] != 'validation' or scores['test_read'] or scores['manifest_sha256'] != c['val_sha256']:
        raise ValueError('Candidate cache is not the bound validation split')
    if m.metric_report(scores['rows'], scores['logits'], scores['threshold'], c['min_eval_support']) != verified['validation']:
        raise ValueError('Saved-score replay mismatch')
    baseline = Path(b['refs']['H0_run'])
    old = read(baseline / 'validation_scores_PRIVATE.json')
    if old['rows'] != scores['rows']:
        raise ValueError('Paired validation rows differ')
    for x, y in zip(history, read(baseline / 'history.json')):
        for key in ('steps', 'unique_originals', 'originals_once_each', 'original_real_pairs',
                    'auxiliary_schedule', 'auxiliary_planned_draws', 'main_nuisance_counts',
                    'main_nuisance_by_axis_group_class', 'aux_nuisance_events'):
            if x[key] != y[key]:
                raise ValueError('Native exposure plan differs: ' + key)
    updates = read(root.parent / (root.name + '_UPDATE_SUMMARY.json'))
    if updates['steps'] != sum(r['steps'] for r in history) or updates['mode'] != 'projected':
        raise ValueError('Optimizer-update count differs')
    result = m.bootstrap(scores['rows'], scores['logits'], scores['threshold'], c['min_eval_support'], 1000, 17,
                         other=(old['logits'], old['threshold']))
    for value in result['intervals'].values():
        if value['valid_replicates'] < 900:
            value['ci95'] = None
            value['status'] = 'INCONCLUSIVE_SUPPORT'
    primary = result['intervals']['gender_race/F_FPR']
    favorable = (primary['estimate'] < 0 and result['intervals']['overall/auc']['estimate'] >= 0
                 and result['intervals']['overall/fnr']['estimate'] <= 0)
    supported = primary['valid_replicates'] >= 900
    state = 'NO_EFFECTIVE_PROJECTION' if updates['projected_steps'] == 0 else 'INCONCLUSIVE_PRIMARY_UNCERTAINTY' if not supported else 'POINT_SCREEN_PASSED_REQUIRES_CONFIRMATION' if favorable else 'MIXED_OR_POINT_SCREEN_FAILED'
    out = m.fresh_dir(out)
    m.write_json(out / 'RESULT.json', {'status': state, 'candidate': verified, 'history': history,
                 'control': read(baseline / 'RUN_VERIFIED.json'), 'paired_contrast': result,
                 'point_screen_passed': favorable, 'primary_uncertainty_supported': supported,
                 'update_summary': updates, 'test_cal_read': False,
                 'source_standardized_claim': False, 'formal_superiority_claim': False,
                 'fixed05_candidate': m.metric_report(scores['rows'], scores['logits'], 0., c['min_eval_support']),
                 'fixed05_control': m.metric_report(old['rows'], old['logits'], 0., c['min_eval_support'])})
    lines = ['# Alignment-gradient pilot — FF++ validation only', '', 'Status: **' + state + '**.', '',
             '| Model | Selected epoch | AUC % | FNR % | Gender 100F_FPR | Race | Joint | Age | Skin |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, record in [('H0', read(baseline / 'RUN_VERIFIED.json')), ('H_GP', verified)]:
        vr = record['validation']
        vals = [vr['overall']['auc'], vr['overall']['fnr']] + [vr['axes'][a]['F_FPR'] for a in ('gender', 'race', 'gender_race', 'age_bin', 'skin_bin')]
        lines.append('|' + name + '|' + str(record['selected_epoch']) + '|' + '|'.join(f'{v*100:.6f}' if v is not None else 'unavailable' for v in vals) + '|')
    lines += ['', 'Both models use the same selection policy and their own stored global VAL10 threshold.',
              'RESULT.json retains per-epoch histories and all group/family/axis metrics, regressions, paired contrasts and valid-draw counts.',
              'Unsupported intervals are withheld. One seed and exposed source validation do not establish superiority or formal noninferiority.',
              'No TEST/CAL access or automatic extension.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': state, 'point_screen_passed': favorable, 'primary': primary}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    q = sub.add_parser('bind'); q.add_argument('--references', required=True); q.add_argument('--out', required=True)
    q = sub.add_parser('diagnose'); q.add_argument('--binding', required=True); q.add_argument('--out', required=True); q.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    q = sub.add_parser('software-check'); q.add_argument('--out', required=True); q.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    q = sub.add_parser('profile')
    for name in ('binding', 'diagnostic', 'software', 'out'): q.add_argument('--' + name, required=True)
    q.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    q = sub.add_parser('train')
    for name in ('binding', 'diagnostic', 'software', 'profile', 'out'): q.add_argument('--' + name, required=True)
    q.add_argument('--budget-hours', required=True, type=float)
    q = sub.add_parser('report')
    for name in ('binding', 'candidate', 'out'): q.add_argument('--' + name, required=True)
    a = parser.parse_args(); torch.set_num_threads(2)
    if a.command == 'bind': bind(a.references, a.out)
    elif a.command == 'diagnose': diagnose(a.binding, a.out, torch.device(a.device))
    elif a.command == 'software-check': software_check(a.out, torch.device(a.device))
    elif a.command == 'profile': profile(a.binding, a.diagnostic, a.software, a.out, torch.device(a.device))
    elif a.command == 'train': train_candidate(a.binding, a.diagnostic, a.software, a.profile, a.out, a.budget_hours)
    else: report(a.binding, a.candidate, a.out)


if __name__ == '__main__':
    main()
