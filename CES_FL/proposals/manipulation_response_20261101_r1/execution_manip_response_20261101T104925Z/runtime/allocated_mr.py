"""One explicitly selected milestone inside PBS ngpus=1; never chains stages."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/manip_response_20261101T104925Z')
SOURCE = Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb')
COMMIT = '58773bbef9d9fe79dd313610f694ddeea12c6037'
GIT = '/home/sachin.chaudhary/ces_fl_tools/git_cluster_20261031/bin/git'
QWEN = Path('/home/sachin.chaudhary/ces_fl_runs/qwen_quality32_20261101T095157Z')
PROJECT_LOCKS = Path('/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA/work/gpulocks')
CONFIG = SOURCE / 'CES_FL/proposals/manipulation_response_20261101_r1/config.json'
SCRIPT = SOURCE / 'CES_FL/src/ces_fl_manipulation_response.py'
PYTHON = '/home/sachin.chaudhary/.conda/envs/aiface/bin/python'


def capture(argv):
    return subprocess.check_output(argv, text=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def normalized_uuid(value):
    return str(uuid.UUID(value[4:] if value.upper().startswith('GPU-') else value))


def validate_allocation(record, host):
    if record.get('job_state') != 'R' or int(record.get('Resource_List', {}).get('ngpus', 0)) != 1:
        raise ValueError('Requires RUNNING PBS ngpus=1 allocation')
    hosts = {part.split('/')[0] for part in record.get('exec_host', '').split('+')}
    if hosts != {host} or int(record.get('Resource_List', {}).get('nodect', 1)) != 1:
        raise ValueError('Allocation does not match this single host')


def eligible(row, occupied, protected, permitted=None):
    index, device_uuid, total, free, utilization = row
    return (device_uuid not in occupied and device_uuid not in protected
            and int(free) >= 24576 and int(utilization) <= 10
            and (permitted is None or index in permitted or device_uuid in permitted))


def verified_sources():
    assert capture([GIT, '-C', str(SOURCE), 'rev-parse', 'HEAD']).strip() == COMMIT
    assert subprocess.run([GIT, '-C', str(SOURCE), 'symbolic-ref', '-q', 'HEAD'],
                          capture_output=True).returncode == 1
    assert not capture([GIT, '-C', str(SOURCE), 'status', '--porcelain', '--', 'CES_FL']).strip()
    expected = json.loads((SOURCE / 'CES_FL/proposals/manipulation_response_20261101_r1/evidence/source_manifest.json').read_text())
    for rel, digest in expected.items():
        assert sha(SOURCE / rel) == digest, 'Changed source/config/test: ' + rel
    return expected


def main(stage):
    assert stage in ('smoke', 'real_preflight', 'P', 'bce', 'fair')
    job = os.environ['PBS_JOBID']
    host = socket.gethostname().split('.')[0]
    hashes = verified_sources()
    allocation_raw = json.loads(capture(['qstat', '-f', '-F', 'json', job]))['Jobs'][job]
    validate_allocation(allocation_raw, host)
    allocation = {k: allocation_raw[k] for k in ('Job_Name', 'job_state', 'queue', 'exec_host', 'exec_vnode', 'Resource_List') if k in allocation_raw}
    write(ROOT / 'checks' / (stage + '_ALLOCATION.json'), allocation)
    if (ROOT / 'checks' / (stage + '_PASS.json')).exists() or (ROOT / 'checks' / (stage + '_FAILURE.json')).exists():
        raise ValueError('Prior milestone output is immutable; no implicit rerun')
    if stage != 'smoke':
        smoke = json.loads((ROOT / 'smoke_cuda/RESULT.json').read_text())
        assert smoke['status'] == 'PASS' and smoke['device'] == 'cuda'
        assert smoke['hashes']['trainer'] == hashes['CES_FL/src/ces_fl_manipulation_response.py']
        assert smoke['test_sha256'] == hashes['CES_FL/tests/smoke_manipulation_response.py']
        assert (ROOT / 'checks/smoke_PASS.json').is_file()
    if stage in ('P', 'bce', 'fair'):
        assert (ROOT / 'checks/real_preflight_PASS.json').is_file()
    guard = (ROOT / 'runtime' / (stage + '.lock')).open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease = registry = child = None
    owner = {'job': job, 'root': str(ROOT), 'stage': stage, 'pid': os.getpid()}

    def stop(number, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, number)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise SystemExit(128 + number)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    try:
        # This site records the allocated host/count, not a UUID. Bind an EMPTY
        # physical device inside that allocation, respecting cooperative locks.
        deadline = time.monotonic() + 7200
        while lease is None:
            protected = set()
            qbinding = QWEN / 'checks/pilot_BINDING.json'
            if qbinding.exists() and not any((QWEN / 'checks' / n).exists() for n in ('pilot_PASS.json', 'pilot_FAILURE.json')):
                q = json.loads(qbinding.read_text())
                if q['host'] == host:
                    protected.add(q['gpu_uuid'])
            occupied = set(capture(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader']).splitlines())
            raw = capture(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'])
            devices = [[v.strip() for v in r] for r in csv.reader(raw.splitlines())]
            inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
            permitted = set(inherited.split(',')) if inherited else None
            for row in sorted(devices, key=lambda r: (-int(r[3]), int(r[4]))):
                if not eligible(row, occupied, protected, permitted):
                    continue
                index, gpu_uuid, total, free, utilization = row
                registry = Path('/tmp/scf_gpu_locks') / ('gpu' + index)
                registry.parent.mkdir(exist_ok=True)
                try:
                    registry.mkdir()
                except FileExistsError:
                    registry = None
                    continue
                (registry / 'cesfl_owner.json').write_text(json.dumps(owner))
                candidate = PROJECT_LOCKS / (host + '_' + index)
                try:
                    candidate.mkdir()
                except FileExistsError:
                    (registry / 'cesfl_owner.json').unlink(); registry.rmdir(); registry = None
                    continue
                lease = candidate
                (lease / 'by').write_text(json.dumps(owner))
                time.sleep(2)
                current = set(capture(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader']).splitlines())
                refreshed_free = int(capture(['nvidia-smi', '-i', index, '--query-gpu=memory.free', '--format=csv,noheader,nounits']).strip())
                if gpu_uuid in current or refreshed_free < 24576:
                    (lease / 'by').unlink(); lease.rmdir(); lease = None
                    (registry / 'cesfl_owner.json').unlink(); registry.rmdir(); registry = None
                    continue
                free = str(refreshed_free)
                break
            if lease is None:
                write(ROOT / 'checks' / (stage + '_PROGRESS.json'), {'phase': 'WAITING_FOR_EMPTY_GPU_IN_ALLOCATION', 'job': job, 'host': host,
                      'protected_qwen_uuids': sorted(protected), 'no_existing_job_modified': True})
                if time.monotonic() >= deadline:
                    raise RuntimeError('No empty unreserved GPU within allocated host after two hours; no job preempted')
                time.sleep(10)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu_uuid, 'PYTHONDONTWRITEBYTECODE': '1',
               'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
        probe = json.loads(capture_with_env([PYTHON, '-c', "import torch,json; assert torch.cuda.is_available() and torch.cuda.device_count()==1; p=torch.cuda.get_device_properties(0); print(json.dumps({'uuid':str(p.uuid),'device':p.name,'torch':str(torch.__version__)}))"], env))
        assert normalized_uuid(probe['uuid']) == normalized_uuid(gpu_uuid)
        binding = {'job': job, 'host': host, 'physical_index': int(index), 'gpu_uuid': gpu_uuid,
                   'free_mib_before_launch': int(free), 'torch_binding': probe, 'source_commit': COMMIT,
                   'source_hashes': hashes, 'operator_sha256': sha(__file__), 'registry': str(registry),
                   'method': 'PBS ngpus=1 on allocated host; site gives count not UUID; empty physical UUID verified under shared registry/project leases',
                   'pbs_assigned_uuid_claimed': False, 'qwen_preempted': False}
        write(ROOT / 'checks' / (stage + '_BINDING.json'), binding)
        if stage == 'smoke':
            command = [PYTHON, str(SOURCE / 'CES_FL/tests/smoke_manipulation_response.py'), '--device', 'cuda', '--out', str(ROOT / 'smoke_cuda')]
        elif stage == 'real_preflight':
            command = [PYTHON, str(ROOT / 'runtime/real_inputs.py')]
        elif stage == 'P':
            command = [PYTHON, str(SCRIPT), 'pretrain', '--config', str(CONFIG), '--out', str(ROOT / 'P'), '--device', 'cuda']
        else:
            bound = ROOT / 'bound_originals/bound_config.json'
            assert bound.is_file(), 'Completed P must be explicitly bound first'
            assert (ROOT / ('dry_' + stage) / 'DRY_RUN.json').is_file(), 'Configured arm backward check missing'
            command = [PYTHON, str(SCRIPT), 'train', '--config', str(bound), '--arm', stage, '--out', str(ROOT / stage), '--device', 'cuda']
        write(ROOT / 'checks' / (stage + '_PROGRESS.json'), {'phase': 'RUNNING', 'job': job, 'host': host, 'command': command, 'gpu_uuid': gpu_uuid})
        with (ROOT / 'logs' / (stage + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        if code:
            raise RuntimeError(stage + ' child exited ' + str(code) + '; later milestones stopped')
        if stage == 'smoke':
            result = json.loads((ROOT / 'smoke_cuda/RESULT.json').read_text())
            assert result['status'] == 'PASS' and result['device'] == 'cuda' and len(result['checks']) == 6
            assert result['hashes']['trainer'] == hashes['CES_FL/src/ces_fl_manipulation_response.py']
            assert result['test_sha256'] == hashes['CES_FL/tests/smoke_manipulation_response.py']
        write(ROOT / 'checks' / (stage + '_PASS.json'), {'status': 'PASS', 'job': job, 'stage': stage,
              'elapsed_seconds': time.monotonic() - started, 'binding': binding, 'command': command,
              'next_milestone_launched': False})
        write(ROOT / 'checks' / (stage + '_PROGRESS.json'), {'phase': 'COMPLETED', 'job': job, 'host': host})
    finally:
        if lease is not None:
            assert json.loads((lease / 'by').read_text()) == owner
            (lease / 'by').unlink(); lease.rmdir()
        if registry is not None:
            assert json.loads((registry / 'cesfl_owner.json').read_text()) == owner
            (registry / 'cesfl_owner.json').unlink(); registry.rmdir()


def capture_with_env(argv, env):
    return subprocess.check_output(argv, text=True, env=env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['smoke', 'real_preflight', 'P', 'bce', 'fair'], required=True)
    args = parser.parse_args()
    try:
        main(args.stage)
    except BaseException as exc:
        write(ROOT / 'checks' / (args.stage + '_FAILURE.json'), {'stage': args.stage, 'exception': repr(exc),
              'observed_node_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'pbs_jobid': os.environ.get('PBS_JOBID'), 'later_milestones_launched': False})
        raise
