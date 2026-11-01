"""Instrument 32-source quality-only generation; never approves edits or trains."""
import datetime
import hashlib
import importlib.metadata
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/qwen_quality32_20261101T095157Z')
SOURCE = Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-cf5050-fc4e63b')
SCRIPT = SOURCE / 'CES_FL/src/ces_fl_counterfactual_5050.py'
CONFIG = SOURCE / 'CES_FL/proposals/counterfactual_5050_20261101_r1/protocol_config.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    path = ROOT / 'checks' / name
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    assert (ROOT / 'checks/preflight_PASS.json').is_file()
    assert sha(SCRIPT) == '3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab'
    assert sha(CONFIG) == '56e77211a966ae12a449c47cdaf8e3fc271f149c6031513747813a1d1abfaef0'
    config = json.loads(CONFIG.read_text())
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    meminfo = Path('/proc/meminfo').read_text().splitlines()
    available = int(next(x for x in meminfo if x.startswith('MemAvailable:')).split()[1]) * 1024
    assert available >= 160 * 2**30, 'Insufficient CPU RAM headroom for the cached Qwen model'
    snapshot = Path(config['model_snapshot'])
    started = time.monotonic()
    inventory = []
    print('MODEL_BINARY_HASH_INVENTORY_START', str(snapshot), flush=True)
    for path in sorted(snapshot.rglob('*')):
        if path.is_file():
            before = path.stat()
            digest = sha(path)
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Model file changed during hashing'
            inventory.append({'path': str(path.relative_to(snapshot)), 'bytes': after.st_size, 'sha256': digest})
    assert any(x['path'].endswith('.safetensors') for x in inventory)
    fingerprint = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    write('MODEL_BINARY_INVENTORY.json', {'snapshot': str(snapshot), 'files': len(inventory),
          'total_bytes': sum(x['bytes'] for x in inventory), 'inventory_sha256': fingerprint,
          'file_inventory': inventory, 'elapsed_seconds': time.monotonic() - started})
    print('MODEL_BINARY_HASH_INVENTORY_PASS', fingerprint, flush=True)
    assert fingerprint == '7ad7899d5af6a6f4f4550d10024ebdb07da83ba2b7b4c1775ef82066f8f790d9', 'Pinned model bytes changed'
    versions = {x: importlib.metadata.version(x) for x in ['torch', 'diffusers', 'transformers', 'accelerate', 'safetensors', 'numpy', 'Pillow', 'scikit-learn', 'scipy']}
    versions['python_executable'] = sys.executable
    torch.cuda.reset_peak_memory_stats()
    generation_start = time.monotonic()
    failure = None
    try:
        plan = json.loads((ROOT / 'generation/plan.json').read_text())
        assert plan['quality_only'] and plan['full_train_binding_prohibited'] and plan['source_count'] == 32
        assert plan['config'] == config
        assert json.loads((ROOT / 'checks/SELECTION_TESTS.json').read_text())['status'] == 'PASS'
        sys.argv = [str(SCRIPT), 'generate', '--plan', str(ROOT / 'generation')]
        runpy.run_path(str(SCRIPT), run_name='__main__')
    except Exception as exc:
        failure = {'exception': str(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        records = []
        for path in sorted((ROOT / 'generation/items').glob('*/record.json')):
            record = json.loads(path.read_text())
            for condition, v in record['variants'].items():
                assert not v['approved'] and v['input_sha256'] == record['source']['original_sha256']
                records.append({'condition': condition, 'seconds': v['seconds'], 'bytes': Path(v['image']).stat().st_size,
                                'label': record['source']['label'], 'independent_from_original': v['independent_from_original']})
        item_bytes = sum(p.stat().st_size for p in (ROOT / 'generation/items').rglob('*') if p.is_file())
        profile = {'status': 'FAIL' if failure else 'PASS', 'observed_node_time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'versions': versions, 'source_sha256': sha(SCRIPT), 'config_sha256': sha(CONFIG),
                   'snapshot_inventory_sha256': fingerprint, 'generation_call_seconds': time.monotonic() - generation_start,
                   'outputs_recorded': len(records), 'output_records_aggregate': records,
                   'mean_seconds_per_output_excluding_load': sum(x['seconds'] for x in records) / len(records) if records else None,
                   'mean_canonical_png_bytes': sum(x['bytes'] for x in records) / len(records) if records else None,
                   'pilot_item_logical_bytes': item_bytes, 'peak_torch_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'peak_torch_reserved_bytes': torch.cuda.max_memory_reserved(), 'allocator_fraction': os.environ.get('CES_CUDA_MEMORY_FRACTION'),
                   'cpu_memory_available_before_load': available, 'execution_mode': 'direct_ssh_bounded_shared_gpu',
                   'full_training_started': False, 'quality_only_sources': 32, 'new_training_authorized': False, 'human_approvals_created': 0,
                   'forensic_or_identity_validity_established': False, 'failure': failure}
        if not failure:
            assert len(records) == 224 and sum(x['label'] == 0 for x in records) == 112 and sum(x['label'] == 1 for x in records) == 112
            profile['historical_full_bank_projection_gpu_hours_not_authorized'] = 532973 * profile['mean_seconds_per_output_excluding_load'] / 3600
            profile['projection_limitations'] = '32 selected sources on a shared GPU; quality only, no valid labels inferred; excludes review/retries and does not authorize scaling.'
        write('PILOT_PROFILE.json', profile)
        print('PILOT_PROFILE', json.dumps({k: v for k, v in profile.items() if k not in ('failure', 'output_records_aggregate')}), flush=True)


if __name__ == '__main__':
    main()
