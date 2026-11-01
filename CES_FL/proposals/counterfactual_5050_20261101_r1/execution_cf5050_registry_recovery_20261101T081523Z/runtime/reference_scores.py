"""CPU inference diagnostic only; never selects edits or updates a model."""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    assert (ROOT / 'checks/pilot_PASS.json').exists()
    plan = json.loads((ROOT / 'checks/REFERENCE_DETECTOR_PLAN.json').read_text())
    source = Path(plan['source'])
    for rel, expected in plan['source_sha256'].items():
        assert sha(source / rel) == expected
    assert sha(plan['checkpoint']) == plan['checkpoint_sha256']
    sys.path.insert(0, str(source / 'CES_FL/src'))
    import ces_fl_upgrade as up
    torch.set_num_threads(4)
    model, checkpoint = up.load_model(plan['checkpoint'], 'cpu')
    before = state_hash(model)
    calibration = dict(checkpoint['calibration'])
    rows, identities = [], []
    for i, path in enumerate(sorted((ROOT / 'generation/items').glob('*/record.json'))):
        record = json.loads(path.read_text())
        source = record['source']
        entries = [('original', source['original_image'])] + [(condition, value['image']) for condition, value in record['variants'].items()] + [('resize_geometry', record['resize_control']['image'])]
        for condition, file in entries:
            rows.append({'image': file, 'label': source['label'], 'identity_ids': source['identity_ids'], 'source_ids': source['source_ids']})
            identities.append({'source_index': i, 'source_label': source['label'], 'condition': condition, 'image': file})
    assert len(rows) == 18
    scores = up.base.predict(model, rows, 'cpu', 2, 0, 224)
    assert len(scores) == len(rows) and all(math.isfinite(float(x)) for x in scores)
    assert before == state_hash(model) and all(p.grad is None for p in model.parameters())
    assert calibration == checkpoint['calibration'] and sha(plan['checkpoint']) == plan['checkpoint_sha256']
    output = []
    originals = {}
    controls = {}
    for info, score in zip(identities, scores):
        z = float(score)
        row = {**info, 'fake_logit': z, 'fake_sigmoid_score': 1 / (1 + math.exp(-z)),
               'source_label_margin': (2 * info['source_label'] - 1) * z,
               'generated_label_is_ground_truth': False}
        output.append(row)
        if info['condition'] == 'original':
            originals[info['source_index']] = row
        if info['condition'] == 'nochange':
            controls[info['source_index']] = row
    for row in output:
        row['fake_score_shift_from_original'] = row['fake_sigmoid_score'] - originals[row['source_index']]['fake_sigmoid_score']
        row['fake_score_shift_from_nochange'] = row['fake_sigmoid_score'] - controls[row['source_index']]['fake_sigmoid_score']
    out = ROOT / 'pilot_inspection'
    assert out.is_dir()
    with (out / 'reference_detector_private.json').open('x') as f:
        json.dump({'reference': plan, 'calibration_unchanged': calibration, 'model_unchanged': True,
                   'scores': output, 'approval_or_rejection_rule': False,
                   'limits': 'Score shifts do not establish counterfactual validity or fairness; no test data or gradients used.'}, f, indent=2, allow_nan=False)
        f.write('\n')
    print('REFERENCE_CPU_DIAGNOSTIC_PASS', len(output), 'images; model/checkpoint/threshold unchanged; approvals=0')


if __name__ == '__main__':
    main()
