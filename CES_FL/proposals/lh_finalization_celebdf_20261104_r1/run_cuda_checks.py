#!/usr/bin/env python3
"""PBS-only CUDA smoke and real-input backward check; performs no optimizer step."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[3]
ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/lh_finalization_celebdf_20261104_43c605b')
OUT = ROOT / 'cuda_checks_352e837'
LEDGER_MODULE = Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/allocated_stage.py')
SESSION = 'lh_resnet50_cuda_checks_352e837'
PYTHON = '/home/sachin.chaudhary/.conda/envs/aiface/bin/python'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not os.environ.get('PBS_JOBID'):
        raise RuntimeError('PBS_JOBID missing; this check requires an actual allocation')
    if OUT.exists():
        raise RuntimeError(f'Preserve existing output directory: {OUT}')
    OUT.mkdir(parents=True)
    qstat = subprocess.check_output(['/opt/pbs/bin/qstat', '-f', os.environ['PBS_JOBID']], text=True, timeout=30)
    if 'Resource_List.ngpus = 1' not in qstat:
        raise RuntimeError('PBS does not confirm exactly one requested GPU')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not visible:
        gpu_file = os.environ.get('PBS_GPUFILE', '').strip()
        if not gpu_file or not Path(gpu_file).is_file():
            raise RuntimeError('No CUDA_VISIBLE_DEVICES or PBS_GPUFILE binding supplied by PBS')
        entries = [line.strip() for line in Path(gpu_file).read_text().splitlines() if line.strip()]
        if len(entries) != 1:
            raise RuntimeError('PBS_GPUFILE must identify exactly one GPU')
        if entries[0].isdigit() or entries[0].startswith('GPU-'):
            visible = entries[0]
        else:
            raise RuntimeError('PBS_GPUFILE entry format is unrecognized; refusing to guess a device')
        os.environ['CUDA_VISIBLE_DEVICES'] = visible

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('CUDA binding does not expose exactly one device')
    if 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError(f'Unexpected allocated device: {torch.cuda.get_device_name(0)}')

    budget = load('cesfl_shared_budget', LEDGER_MODULE)
    started = time.monotonic()
    with budget.ledger() as data:
        session = next(s for s in data['gpu_sessions'] if s['id'] == SESSION)
        if session['state'] != 'QUEUED' or session['ngpus'] != 1:
            raise RuntimeError('Missing unique queued one-GPU reservation')
        session.update(state='RUNNING', pbs_job_id=os.environ['PBS_JOBID'],
                       host=os.uname().nodename, pid=os.getpid(), device=torch.cuda.get_device_name(0),
                       cuda_visible_devices=visible)
    status = 'FAILED'
    try:
        proposal = SOURCE / 'CES_FL/proposals/lh_finalization_celebdf_20261104_r1'
        p50 = load('cesfl_resnet50_p50', SOURCE / 'CES_FL/src/ces_fl_resnet50_p50.py')
        head = load('cesfl_head_resnet50', SOURCE / 'CES_FL/src/ces_fl_head_ablation_resnet50.py')
        config = json.loads((proposal / 'configs/resnet50_p50.json').read_text())
        if p50.sha(p50.__file__) != '2d66966b1733feec222fe3a80ff9a6cd7181a6ba28cb9f767ad980968d337098':
            raise RuntimeError('P50 source hash changed')
        if head.sha(head.__file__) != '72d368a439658c601a5fe4d717237b1561190e85c13c70f6db3a441bc5cdbe78':
            raise RuntimeError('Downstream source hash changed')
        p50.validate_config(config)
        qstat_path = OUT / 'qstat_PRIVATE.txt'
        qstat_path.write_text(qstat)
        result = {
            'status': 'RUNNING', 'commit': subprocess.check_output(['/home/soft/anaconda3/pkgs/git-2.45.2-pl5340h9abc3c3_1/bin/git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
            'pbs_job_id': os.environ['PBS_JOBID'], 'hostname': os.uname().nodename,
            'ngpus': 1, 'cuda_visible_devices': visible, 'device_name': torch.cuda.get_device_name(0),
            'source_sha256': {'p50': p50.sha(p50.__file__), 'downstream': head.sha(head.__file__)},
            'config_sha256': sha(proposal / 'configs/resnet50_p50.json'),
            'manifest_sha256': {'train': sha(config['train']), 'val': sha(config['val']),
                                'overlap': sha(config['protocol_overlap_file']), 'imagenet_weights': sha(config['weights'])},
            'test_read': False, 'optimizer_steps': 0
        }
        (OUT / 'BINDING.json').write_text(json.dumps(result, indent=2) + '\n')
        p50.smoke('cuda', OUT / 'smoke_p50')
        head.smoke(torch.device('cuda'), OUT / 'smoke_downstream', 'linear')
        dry = p50.dry_run(config, 'pretrain', OUT / 'real_input_pretrain_dry', 'cuda')
        if dry.get('status') != 'PASS' or dry.get('optimizer_steps') != 0:
            raise RuntimeError('Real-input backward check failed or updated parameters')
        result.update(status='PASS', synthetic_p50_smoke='PASS', synthetic_downstream_smoke='PASS',
                      real_input_backward='PASS', real_input_optimizer_steps=0,
                      elapsed_seconds=time.monotonic() - started, test_read=False)
        (OUT / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
        status = 'COMPLETED'
    finally:
        elapsed = time.monotonic() - started
        with budget.ledger() as data:
            session = next(s for s in data['gpu_sessions'] if s['id'] == SESSION)
            session.update(state=status, elapsed_seconds=elapsed)
        (OUT / 'ALLOCATION_END.json').write_text(json.dumps({
            'status': status, 'pbs_job_id': os.environ['PBS_JOBID'],
            'elapsed_seconds': elapsed, 'gpu_hours': elapsed / 3600,
            'test_read': False, 'training_launched': False
        }, indent=2) + '\n')


if __name__ == '__main__':
    main()
