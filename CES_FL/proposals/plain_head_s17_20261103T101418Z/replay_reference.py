#!/usr/bin/env python3
"""Replay only the fixed VAL split of completed gated A/H; never train or read TEST."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
CES = HERE.parents[1]
spec = importlib.util.spec_from_file_location('head_model', CES / 'src/ces_fl_head_ablation.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def reference_config(reference):
    return json.loads((CES / ('results/forensic_response_recovery_c332fd34_20261103/ffpp_bound_config.json'
        if reference == 'A' else 'proposals/hierarchical_real_risk_s17_20261103_r1/H_joint_real_bound.json')).read_text())


def verify_run(ck, receipt, history, expected, reference, checkpoint_hash):
    arm = 'A' if reference == 'A' else 'D'
    if ck['config'] != expected or ck['format'] != expected['format'] or ck['source_sha256'] != expected['source_sha256']:
        raise ValueError('Reference config/source/format mismatch')
    if ck['arm'] != arm or ck.get('synthetic_fixture') is not False:
        raise ValueError('Wrong reference arm or fixture')
    if receipt['status'] != 'COMPLETED_VERIFIED' or receipt['epochs_completed'] != 10 or receipt['arm'] != arm:
        raise ValueError('Reference is not completed/verified')
    if receipt['source_sha256'] != expected['source_sha256'] or receipt['config_digest'] != m.digest(expected):
        raise ValueError('Reference receipt binding mismatch')
    if receipt['selected_checkpoint_sha256'] != checkpoint_hash or receipt['validation'] != ck['validation']:
        raise ValueError('Reference checkpoint changed')
    if [r['epoch'] for r in history] != list(range(1, 11)):
        raise ValueError('Incomplete reference history')
    best = max(history, key=lambda r: r['validation']['overall']['auc'])
    if ck['epoch'] != best['epoch'] or receipt['selected_epoch'] != best['epoch'] or best['validation'] != ck['validation'] or best['calibration'] != ck['calibration']:
        raise ValueError('Reference selection/calibration mismatch')


def replay(reference, run, out, device):
    root = Path(run)
    if not (root / 'DONE').is_file():
        raise ValueError('Reference has no DONE; do not replay an active run')
    c = reference_config(reference)
    local = json.loads((HERE / 'L_A_plain_bce.json').read_text())
    if m.sha(m.__file__) != local['source_sha256']:
        raise ValueError('Ablation implementation changed')
    checkpoint = root / 'best.pt'
    before = m.sha(checkpoint)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    receipt = json.loads((root / 'RUN_VERIFIED.json').read_text())
    history = json.loads((root / 'history.json').read_text())
    verify_run(ck, receipt, history, c, reference, before)
    if m.sha(c['val']) != c['val_sha256']:
        raise ValueError('Fixed VAL manifest changed')
    rows = m.read_manifest(c['val'])
    membership = {r['image']: r for r in ck['fit_membership']}
    for row in rows:
        row['image_sha256'] = m.sha(row['image'])
        if any(row[k] != membership.get(row['image'], {}).get(k) for k in ('image', 'image_sha256', 'identity_ids', 'source_ids')):
            raise ValueError('VAL image/provenance changed')
    model = m.Detector(c['hidden'], c['gate_hidden'], c['encoder_policy'], head_type='gated')
    model.load_state_dict(ck['model'], strict=True)
    model = model.to(device).eval()
    z = m.predict(model, rows, c, device)
    threshold = ck['calibration']['threshold_logit']
    report = m.metric_report(rows, z, threshold, c['min_eval_support'])
    if report != ck['validation'] or m.sha(checkpoint) != before:
        raise ValueError('Reference VAL replay mismatch; preserve and investigate, never reselect')
    out = m.fresh_dir(out)
    m.write_json(out / 'validation_scores_PRIVATE.json', {
        'rows': rows, 'logits': z.tolist(), 'threshold': threshold,
        'min_support': c['min_eval_support'], 'manifest_sha256': c['val_sha256'],
        'checkpoint_sha256': before, 'calibration_source': 'selected checkpoint validation only',
        'split': 'validation', 'test_read': False})
    m.write_json(out / 'VAL_REPLAY.json', {
        'status': 'PASS', 'reference': reference, 'checkpoint_sha256': before,
        'source_sha256': m.sha(m.__file__), 'helper_sha256': m.sha(__file__),
        'validation': report, 'fixed_probability_0_5': m.metric_report(rows,z,0.,c['min_eval_support']),
        'threshold_recalibrated': False, 'test_read': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', choices=['A','H'], required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    replay(a.reference, a.run, a.out, torch.device(a.device))
