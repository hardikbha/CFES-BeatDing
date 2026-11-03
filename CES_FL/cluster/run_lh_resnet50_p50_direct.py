#!/usr/bin/env python3
"""Run the reviewed seed-17 ResNet-50 P50 stage on a direct SSH GPU lease."""
import contextlib
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

SOURCE = Path(__file__).resolve().parents[2]
GPU_HOST = 'gpu01'
GPU_UUID = 'GPU-6cc0a5b3-99ca-13e3-5508-a201a06e82d6'
LOCKS = Path('/tmp/scf_gpu_locks')
OLD_RUNS = Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103')
LEDGER_MODULE = OLD_RUNS / 'allocated_stage.py'
SESSION_ID = 'lh_resnet50_P50_s17_20261104'
RUN_ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/lh_resnet50_table3_s17_20261104')
CONFIG = SOURCE / 'CES_FL/proposals/lh_finalization_celebdf_20261104_r1/configs/resnet50_p50.json'
EXPECTED_P50_SHA = '2d66966b1733feec222fe3a80ff9a6cd7181a6ba28cb9f767ad980968d337098'
EXPECTED_HEAD_SHA = '72d368a439658c601a5fe4d717237b1561190e85c13c70f6db3a441bc5cdbe78'
RESERVED_GPU_HOURS = 10.0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    p = Path(path)
    tmp = p.with_name(p.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, p)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def pool_mutex():
    LOCKS.mkdir(exist_ok=True)
    lock_dir = LOCKS / '.pool_mutex'
    deadline = time.monotonic() + 30
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise RuntimeError('Shared GPU lease mutex is busy; no GPU claimed')
            time.sleep(.2)
    try:
        yield
    finally:
        lock_dir.rmdir()


def query_gpu(uuid):
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'
    ], text=True)
    cards = []
    for row in csv.reader(output.splitlines()):
        cards.append({'index': int(row[0]), 'uuid': row[1].strip(),
                      'free_mib': int(row[2]), 'used_mib': int(row[3]),
                      'utilization': int(row[4])})
    matches = [c for c in cards if c['uuid'] == uuid]
    if len(matches) != 1:
        raise RuntimeError('Pinned GPU UUID is absent or ambiguous')
    apps = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
        '--format=csv,noheader,nounits'
    ], text=True)
    active = []
    for row in csv.reader(apps.splitlines()):
        if row and row[0].strip() == uuid:
            active.append([x.strip() for x in row])
    return matches[0], active


def reserve_budget(budget, gpu):
    with budget.ledger() as data:
        if any(s['id'] == SESSION_ID for s in data['gpu_sessions']):
            raise RuntimeError('Duplicate P50 budget session exists; inspect before resuming')
        committed = sum(s['reserved_gpu_hours'] if s['state'] in ('QUEUED', 'RUNNING')
                        else s.get('elapsed_seconds', 0) * s['ngpus'] / 3600
                        for s in data['gpu_sessions'])
        if committed + RESERVED_GPU_HOURS > data['initial_total_gpu_hours']:
            raise RuntimeError('P50 reservation would exceed the extended shared budget')
        data['gpu_sessions'].append({
            'id': SESSION_ID, 'ngpus': 1, 'reserved_gpu_hours': RESERVED_GPU_HOURS,
            'elapsed_seconds': 0., 'state': 'RUNNING', 'route': 'user_authorized_direct_ssh_pool_v2',
            'pbs_allocation_claimed': False, 'host': os.uname().nodename,
            'controller_pid': os.getpid(), 'card': gpu, 'log': str(RUN_ROOT / 'P50.log'),
            'scientific_commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
            'scientific_source_sha256': EXPECTED_P50_SHA,
            'purpose': 'ResNet-50 seed-17 original-only P50; followed by matched L_A/L_H if validated'
        })


def main():
    if os.uname().nodename.split('.')[0] != GPU_HOST:
        raise RuntimeError(f'Expected direct SSH host {GPU_HOST}; found {os.uname().nodename}')
    if SOURCE.resolve() != Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-lh-finalize-exec-caa6dee').resolve():
        raise RuntimeError('Runner is not using its declared immutable checkout')
    git = '/home/soft/anaconda3/pkgs/git-2.45.2-pl5340h9abc3c3_1/bin/git'
    commit = subprocess.check_output([git, '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.run([git, '-C', str(SOURCE), 'symbolic-ref', '-q', 'HEAD'],
                      stdout=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('Execution checkout is not detached')
    if subprocess.check_output([git, '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Execution checkout is dirty')
    if RUN_ROOT.exists():
        raise RuntimeError(f'Preserve existing P50 output directory: {RUN_ROOT}')
    p50 = load('cesfl_resnet50_p50', SOURCE / 'CES_FL/src/ces_fl_resnet50_p50.py')
    head = load('cesfl_head_resnet50', SOURCE / 'CES_FL/src/ces_fl_head_ablation_resnet50.py')
    if p50.sha(p50.__file__) != EXPECTED_P50_SHA or head.sha(head.__file__) != EXPECTED_HEAD_SHA:
        raise RuntimeError('Scientific source hash mismatch')
    config = read_json(CONFIG)
    p50.validate_config(config)
    gpu, apps = query_gpu(GPU_UUID)
    if gpu['free_mib'] < 32_000 or apps:
        raise RuntimeError(f'Pinned GPU is not free enough: card={gpu}, active_compute_apps={apps}')

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUID
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('CUDA binding failed to expose exactly one GPU')
    if torch.cuda.get_device_name(0) != 'NVIDIA H100 80GB HBM3':
        raise RuntimeError(f'Unexpected physical device: {torch.cuda.get_device_name(0)}')

    lease_dir = LOCKS / 'gpu1'
    lease = lease_dir / f'task_{SESSION_ID}.json'
    with pool_mutex():
        if lease.exists():
            raise RuntimeError('Own GPU lease already exists; no duplicate launch')
        if lease_dir.exists():
            owner_path = lease_dir / 'pool_owner.json'
            if not owner_path.exists() or not read_json(owner_path).get('shared'):
                raise RuntimeError('GPU01 device lease is not a shared pool-v2 device')
            active = [read_json(p) for p in lease_dir.glob('task_*.json')]
            active = [e for e in active if os.path.exists(f"/proc/{e.get('pid',-1)}")]
            if active:
                raise RuntimeError(f'GPU01 device already has active leases: {active}')
        else:
            lease_dir.mkdir()
            write_json(lease_dir / 'pool_owner.json', {'shared': True, 'v': 2, 'pid': 0, 'wid': 'pool_v2_shared'})
        write_json(lease, {'name': SESSION_ID, 'wid': SESSION_ID, 'pid': os.getpid(),
                           'pgid': os.getpgrp(), 'cls': 'writer', 'need': 28000,
                           'time': time.time(), 'device_uuid': GPU_UUID, 'commit': commit})

    torch.cuda.set_per_process_memory_fraction(.85, 0)
    budget = load('cesfl_shared_budget', LEDGER_MODULE)
    try:
        reserve_budget(budget, gpu)
    except Exception:
        with pool_mutex():
            if lease.exists() and read_json(lease).get('pid') == os.getpid():
                lease.unlink()
        raise
    started = time.monotonic()
    stop = threading.Event()
    state = {'status': 'FAILED'}
    run_root_created = False

    def heartbeat():
        while not stop.wait(20):
            with budget.ledger() as data:
                row = next(s for s in data['gpu_sessions'] if s['id'] == SESSION_ID)
                row['elapsed_seconds'] = time.monotonic() - started
                row['device'] = torch.cuda.get_device_name(0)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        RUN_ROOT.mkdir(parents=True, exist_ok=False)
        run_root_created = True
        write_json(RUN_ROOT / 'BINDING.json', {
            'status': 'PASS', 'commit': commit, 'host': os.uname().nodename,
            'gpu': gpu, 'active_compute_apps_at_claim': apps,
            'cuda_visible_devices': GPU_UUID, 'device': torch.cuda.get_device_name(0),
            'p50_source_sha256': EXPECTED_P50_SHA, 'downstream_source_sha256': EXPECTED_HEAD_SHA,
            'config_sha256': p50.sha(CONFIG),
            'input_hashes': {name: p50.sha(config[name]) for name in ('train','val','protocol_overlap_file','weights')},
            'test_read': False, 'optimizer_steps': 0
        })
        p50.smoke('cuda', RUN_ROOT / 'checks' / 'synthetic_p50')
        head.smoke(torch.device('cuda'), RUN_ROOT / 'checks' / 'synthetic_downstream', 'linear')
        dry = p50.dry_run(config, 'pretrain', RUN_ROOT / 'checks' / 'real_input_pretrain_dry', 'cuda')
        if dry.get('status') != 'PASS' or dry.get('optimizer_steps') != 0:
            raise RuntimeError('Configured real-input backward check failed or updated parameters')
        write_json(RUN_ROOT / 'checks' / 'REAL_INPUT_BACKWARD.json', dry)
        write_json(RUN_ROOT / 'P50_START.json', {
            'stage': 'pretrain', 'epochs': 50, 'seed': 17, 'checkpoint_selection': 'earliest strict maximum validation AUC',
            'source_sha256': EXPECTED_P50_SHA, 'config_sha256': p50.sha(CONFIG),
            'device_uuid': GPU_UUID, 'device_name': torch.cuda.get_device_name(0),
            'test_read': False, 'launch_commit': commit
        })
        p50.train(config, 'pretrain', RUN_ROOT / 'P50', 'cuda')
        state.update(status='COMPLETED', checkpoint_sha256=p50.sha(RUN_ROOT / 'P50' / 'best.pt'))
        write_json(RUN_ROOT / 'P50_STATUS.json', state)
    finally:
        stop.set()
        thread.join(timeout=20)
        elapsed = time.monotonic() - started
        state.setdefault('status', 'FAILED')
        state['elapsed_seconds'] = elapsed
        state['gpu_hours'] = elapsed / 3600
        if run_root_created:
            write_json(RUN_ROOT / 'ALLOCATION_END.json', state)
        with budget.ledger() as data:
            row = next(s for s in data['gpu_sessions'] if s['id'] == SESSION_ID)
            row.update(state=state['status'], elapsed_seconds=elapsed, log=str(RUN_ROOT / 'P50.log'))
        with pool_mutex():
            if lease.exists() and read_json(lease).get('pid') == os.getpid():
                lease.unlink()


if __name__ == '__main__':
    main()
