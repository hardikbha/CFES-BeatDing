"""Allocated, original-only decode/hash audit followed by the exact backward check."""
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/manip_response_20261101T104925Z')
SOURCE = Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb')
INVENTORY = Path('/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z/generation/sources.json')
sys.path.insert(0, str(SOURCE / 'CES_FL/src'))
import ces_fl_manipulation_response as mr


def inspect(item):
    split, row, expected = item
    path = Path(row['image'])
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Image changed while reading: ' + str(path))
    digest = hashlib.sha256(data).hexdigest()
    if expected and digest != expected:
        raise RuntimeError('TRAIN original differs from pinned inventory: ' + str(path))
    with mr.base.Image.open(io.BytesIO(data)) as image:
        image.load()
        image.convert('RGB').load()
        size = image.size
    return {'split': split, 'image': str(path), 'sha256': digest,
            'bytes': len(data), 'size': size, 'label': row['label']}


def main():
    mr.hashes()
    c = mr.read_config(SOURCE / 'CES_FL/proposals/manipulation_response_20261101_r1/config.json')
    rows, val, weights, splits = mr.base.data_preflight(c['pretrain'])
    assert mr.base.sha(INVENTORY) == '5f8792a7017346b8205fa4a9b67257d31dc50113f2612a02b2f08506004d5810'
    inventory = json.loads(INVENTORY.read_text())
    assert [r['image'] for r in rows] == [s['original_image'] for s in inventory]
    expected = {s['original_image']: s['original_sha256'] for s in inventory}
    out = mr.base.fresh(ROOT / 'real_inputs')
    counts, sizes, label_groups = collections.Counter(), collections.Counter(), collections.Counter()
    hashes = {'train': set(), 'val': set()}
    tasks = [('train', r, expected[r['image']]) for r in rows] + [('val', r, None) for r in val]
    with (out / 'image_inventory_PRIVATE.jsonl').open('x') as f, ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(inspect, tasks):
            f.write(json.dumps(result) + '\n')
            split = result['split']
            counts[split] += 1
            hashes[split].add(result['sha256'])
            sizes[str(tuple(result['size']))] += 1
            if sum(counts.values()) % 2000 == 0:
                print('DECODED_AND_HASHED', dict(counts), flush=True)
    overlap = hashes['train'] & hashes['val']
    report = {'status': 'PASS' if not overlap else 'BLOCKED_EXACT_TRAIN_VAL_IMAGE_BYTES',
              'hashes': mr.hashes(), 'config_sha256': mr.base.sha(SOURCE / 'CES_FL/proposals/manipulation_response_20261101_r1/config.json'),
              'decoded_counts': dict(counts), 'dimensions': dict(sizes), 'split_protocol': splits,
              'exact_train_val_byte_overlap': len(overlap),
              'train_label_counts': dict(collections.Counter(r['label'] for r in rows)),
              'val_label_counts': dict(collections.Counter(r['label'] for r in val)),
              'train_gender_race_counts': dict(collections.Counter(r['gender_race'] for r in rows)),
              'train_label_ids8_counts': {str(k): v for k, v in collections.Counter((r['label'], r['ids8']) for r in rows).items()},
              'private_inventory_sha256': mr.base.sha(out / 'image_inventory_PRIVATE.jsonl'),
              'test_images_read': False, 'optimizer_steps': 0,
              'semantic_authenticity_or_person_identity_verified': False}
    mr.base.write(out / 'DECODE_AUDIT.json', report)
    if overlap:
        raise RuntimeError('Exact TRAIN/VAL image bytes overlap; report before any training, do not silently alter split')
    command = ['/home/sachin.chaudhary/.conda/envs/aiface/bin/python',
               str(SOURCE / 'CES_FL/src/ces_fl_counterfactual_5050.py'), 'dry-run', '--config',
               str(SOURCE / 'CES_FL/proposals/counterfactual_5050_20261101_r1/protocol_config.json'),
               '--stage', 'pretrain', '--device', 'cuda', '--out', str(ROOT / 'P_real_dry')]
    print('EXACT_BACKWARD_COMMAND', json.dumps(command), flush=True)
    subprocess.run(command, check=True, cwd=SOURCE)
    dry = json.loads((ROOT / 'P_real_dry/RESULT.json').read_text())
    assert dry['status'] == 'PASS' and dry['device'] == 'cuda' and dry['optimizer_steps'] == 0
    assert dry['source_sha256'] == mr.BASE_SHA and not dry['synthetic_fixture']
    print('ORIGINAL_INPUT_AND_REAL_BACKWARD_PASS', flush=True)


if __name__ == '__main__':
    main()
