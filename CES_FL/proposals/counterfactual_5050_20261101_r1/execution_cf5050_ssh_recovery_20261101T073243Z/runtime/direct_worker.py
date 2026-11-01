"""Explicit single-milestone SSH worker. No automatic milestone chaining."""
import argparse
import csv
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
import uuid as uuidlib
from pathlib import Path

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/cf5050_ssh_recovery_20261101T073243Z')
SOURCE = Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-cf5050-fc4e63b')
SCRIPT = SOURCE / 'CES_FL/src/ces_fl_counterfactual_5050.py'
CONFIG = SOURCE / 'CES_FL/proposals/counterfactual_5050_20261101_r1/protocol_config.json'
GIT = '/home/sachin.chaudhary/ces_fl_tools/git_cluster_20261031/bin/git'
LOCKS = Path('/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA/work/gpulocks')


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def run_command(stage, name, command, env):
    log = ROOT / 'logs' / (name + '.log')
    print('START', name, json.dumps(command), flush=True)
    started = time.monotonic()
    with log.open('x') as f:
        p = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, cwd=SOURCE, env=env)
    write(ROOT / 'checks' / (name + '_EXIT.json'), {'stage': stage, 'command': command, 'exit_code': p.returncode,
          'elapsed_seconds': time.monotonic() - started, 'log': str(log), 'execution_mode': 'direct_ssh'})
    if p.returncode:
        raise RuntimeError(name + ' failed; preserve its log and stop later stages')
    print('PASS', name, flush=True)


def canonical_uuid(value):
    value = value[4:] if value.upper().startswith('GPU-') else value
    return str(uuidlib.UUID(value))


def main(stage):
    host = socket.gethostname().split('.')[0]
    assert host in ('gpu01', 'gpu02')
    assert not (ROOT / 'checks' / (stage + '_PASS.json')).exists()
    assert not (ROOT / 'checks' / (stage + '_FAILURE.json')).exists()
    assert subprocess.check_output([GIT, '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == 'fc4e63b8811c5813afa0d298a8172ba9555b7e40'
    assert not subprocess.check_output([GIT, '-C', str(SOURCE), 'status', '--porcelain'], text=True)
    baseline = json.loads((ROOT / 'checks/LOCAL_HASH_PREFLIGHT.json').read_text())
    assert sha(SCRIPT) == baseline['source_sha256'] and sha(CONFIG) == baseline['config_sha256']
    for item in baseline['inputs'].values():
        assert sha(item['path']) == item['sha256']
    if stage == 'pilot':
        assert (ROOT / 'checks/preflight_PASS.json').is_file(), 'Real pilot requires successful CUDA/data checks'
        assert (ROOT / 'runtime/pilot_profile.py').is_file()
    fraction, minimum = (0.20, 24576) if stage == 'preflight' else (0.55, 53248)
    lease = None
    started = time.monotonic()
    try:
        while lease is None:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            devices = [[x.strip() for x in row] for row in csv.reader(raw.splitlines())]
            for index, uuid, total, free, util in sorted(devices, key=lambda row: (-int(row[3]), int(row[4]))):
                if int(free) < minimum:
                    continue
                candidate = LOCKS / (host + '_' + index)
                try:
                    candidate.mkdir()
                except FileExistsError:
                    continue
                lease = candidate
                (lease / 'by').write_text(str(ROOT) + '\n' + stage + '\n' + str(os.getpid()) + '\n')
                break
            if lease is None:
                write(ROOT / 'checks' / (stage + '_PROGRESS.json'), {'phase': 'WAITING_FOR_VRAM_HEADROOM', 'host': host,
                      'minimum_free_mib': minimum, 'no_existing_job_modified': True})
                if time.monotonic() - started > 7200:
                    raise RuntimeError('No eligible VRAM headroom within two hours; no existing process was modified')
                time.sleep(20)
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=uuid, CES_CUDA_MEMORY_FRACTION=str(fraction),
                   PYTHONPATH=str(ROOT / 'cuda_policy') + ':' + env.get('PYTHONPATH', ''),
                   PATH=str(Path(GIT).parent) + ':/opt/pbs/bin:/usr/bin:/bin:' + env.get('PATH', ''),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        env.pop('PBS_JOBID', None)
        binding = {'host': host, 'physical_index': int(index), 'gpu_uuid': uuid, 'free_mib_before_launch': int(free),
                   'total_mib': int(total), 'utilization_percent_before_launch': int(util), 'worker_pid': os.getpid(),
                   'cuda_allocator_fraction': fraction, 'execution_mode': 'direct_ssh_bounded_shared_gpu',
                   'pbs_allocation': None, 'exclusive_ownership_claimed': False, 'stage': stage,
                   'source_commit': 'fc4e63b8811c5813afa0d298a8172ba9555b7e40',
                   'source_sha256': sha(SCRIPT), 'config_sha256': sha(CONFIG),
                   'runner_sha256': sha(__file__), 'observed_node_time': time.time(),
                   'preexisting_compute_processes': subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True).splitlines()}
        probe = subprocess.run([sys.executable, '-c', "import torch,json; assert torch.cuda.is_available() and torch.cuda.device_count()==1; p=torch.cuda.get_device_properties(0); print(json.dumps({'device':p.name,'uuid':str(p.uuid),'torch':str(torch.__version__),'total_vram':p.total_memory}))"], env=env, text=True, capture_output=True)
        assert probe.returncode == 0, probe.stderr
        binding['torch_binding'] = json.loads(probe.stdout)
        assert canonical_uuid(binding['torch_binding']['uuid']) == canonical_uuid(uuid), 'Visible CUDA device differs from selected NVIDIA UUID'
        write(ROOT / 'checks' / (stage + '_BINDING.json'), binding)
        write(ROOT / 'checks' / (stage + '_PROGRESS.json'), {'phase': 'RUNNING', **binding})
        if stage == 'preflight':
            commands = [
                ('smoke_cuda', [sys.executable, '-u', str(SCRIPT), 'smoke', '--device', 'cuda', '--out', str(ROOT / 'smoke_cuda')]),
                ('generation_plan', [sys.executable, '-u', str(SCRIPT), 'plan', '--config', str(CONFIG), '--out', str(ROOT / 'generation')]),
                ('dry_pretrain', [sys.executable, '-u', str(SCRIPT), 'dry-run', '--config', str(CONFIG), '--stage', 'pretrain', '--device', 'cuda', '--out', str(ROOT / 'dry_pretrain')]),
            ]
            for name, command in commands:
                write(ROOT / 'checks' / (stage + '_PROGRESS.json'), {'phase': name, 'host': host, 'gpu_uuid': uuid})
                run_command(stage, name, command, env)
            smoke = json.loads((ROOT / 'smoke_cuda/RESULT.json').read_text())
            dry = json.loads((ROOT / 'dry_pretrain/RESULT.json').read_text())
            assert smoke['status'] == 'PASS' and smoke['device'] == 'cuda'
            assert dry['status'] == 'PASS' and dry['optimizer_steps'] == 0 and not dry['synthetic_fixture']
        else:
            run_command(stage, 'qwen_two_source_pilot', [sys.executable, '-u', str(ROOT / 'runtime/pilot_profile.py')], env)
            run_command(stage, 'pilot_review_template', [sys.executable, '-u', str(SCRIPT), 'review-template', '--plan', str(ROOT / 'generation'), '--out', str(ROOT / 'pilot_review_pending.csv')], env)
        write(ROOT / 'checks' / (stage + '_PASS.json'), {'status': 'PASS', 'stage': stage, 'host': host, 'gpu_uuid': uuid,
              'elapsed_seconds': time.monotonic() - started, 'source_commit': binding['source_commit'],
              'source_sha256': sha(SCRIPT), 'config_sha256': sha(CONFIG), 'full_training_started': False,
              'scientific_validity_or_human_approval_implied': False})
        write(ROOT / 'checks' / (stage + '_PROGRESS.json'), {'phase': 'DONE', 'host': host, 'gpu_uuid': uuid})
    finally:
        if lease is not None:
            (lease / 'by').unlink()
            lease.rmdir()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['preflight', 'pilot'], required=True)
    args = parser.parse_args()
    try:
        main(args.stage)
    except Exception as exc:
        write(ROOT / 'checks' / (args.stage + '_FAILURE.json'), {'stage': args.stage, 'exception': str(exc),
              'traceback': traceback.format_exc(), 'host': socket.gethostname(), 'full_training_started': False})
        raise
