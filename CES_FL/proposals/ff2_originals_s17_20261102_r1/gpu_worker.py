"""One explicit SSH stage; respect shared leases, occupied GPUs and previous outputs."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1')
LOCKS = Path('/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA/work/gpulocks')
PYTHON = '/home/sachin.chaudhary/.conda/envs/aiface/bin/python'


def capture(argv):
    return subprocess.check_output(argv, text=True)


def write(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2); f.write('\n')


def main(stage):
    host = socket.gethostname().split('.')[0]
    assert host in ['gpu01', 'gpu02']
    guard = (ROOT / 'runtime' / (stage + '.lock')).open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not any((ROOT / 'checks' / (stage + suffix)).exists() for suffix in ['_PASS.json', '_FAILURE.json', '_EXIT.json'])
    assert (ROOT / 'checks/inputs_PASS.json').exists()
    if stage == 'train':
        assert (ROOT / 'checks/smoke_PASS.json').exists() and not (ROOT / 'training').exists()
    lease = registry = child = None
    owner = {'host': host, 'worker_pid': os.getpid(), 'root': str(ROOT), 'stage': stage}
    start = time.monotonic()
    try:
        while lease is None:
            occupied = set(capture(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader']).splitlines())
            devices = list(csv.reader(capture(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits']).splitlines()))
            for row in sorted(devices, key=lambda x: (-int(x[3]), int(x[0]))):
                index, uuid, total, free, util = [v.strip() for v in row]
                if uuid in occupied or int(free) < 65000 or int(util) > 5:
                    continue
                registry = Path('/tmp/scf_gpu_locks') / ('gpu' + index)
                registry.parent.mkdir(exist_ok=True)
                try:
                    registry.mkdir()
                except FileExistsError:
                    registry = None; continue
                (registry / 'cesfl_owner.json').write_text(json.dumps(owner))
                candidate = LOCKS / (host + '_' + index)
                try:
                    candidate.mkdir()
                except FileExistsError:
                    (registry / 'cesfl_owner.json').unlink(); registry.rmdir(); registry = None; continue
                lease = candidate; (lease / 'by').write_text(json.dumps(owner))
                current = set(capture(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader']).splitlines())
                current_free = int(capture(['nvidia-smi', '-i', uuid, '--query-gpu=memory.free', '--format=csv,noheader,nounits']).strip())
                if uuid in current or current_free < 65000:
                    (lease / 'by').unlink(); lease.rmdir(); lease = None
                    (registry / 'cesfl_owner.json').unlink(); registry.rmdir(); registry = None; continue
                break
            if lease is None:
                if time.monotonic() - start > 600:
                    raise RuntimeError('No idle cooperatively available GPU in10minutes; no jobs modified')
                print('WAITING_FOR_IDLE_COOPERATIVE_GPU', host, flush=True); time.sleep(10)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, PYTHONDONTWRITEBYTECODE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        env.pop('PBS_JOBID', None)
        binding = {**owner, 'physical_index': int(index), 'gpu_uuid': uuid, 'free_mib': current_free,
                   'cuda_memory_fraction': .20, 'execution_mode': 'direct_ssh_cooperative_idle_gpu',
                   'pbs_allocation': None, 'preexisting_processes_stopped': 0}
        write(ROOT / 'checks' / (stage + '_BINDING.json'), binding)
        command = [PYTHON, '-u', str(ROOT / 'runtime/stage_program.py'), '--stage', stage]
        with (ROOT / 'logs' / (stage + '.log')).open('x') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
            write(ROOT / 'checks' / (stage + '_CHILD.json'), {**binding, 'child_pid': child.pid, 'command': command})
            print('CHILD_STARTED', child.pid, binding, flush=True)
            code = child.wait()
        write(ROOT / 'checks' / (stage + '_EXIT.json'), {'exit_code': code, 'elapsed_seconds': time.monotonic() - start, **binding})
        assert code == 0 and (ROOT / 'checks' / (stage + '_PASS.json')).exists(), 'Stage failed; inspect logs; no automatic retry'
    except BaseException as error:
        if not (ROOT / 'checks' / (stage + '_FAILURE.json')).exists():
            write(ROOT / 'checks' / (stage + '_FAILURE.json'), {'error': repr(error), 'traceback': traceback.format_exc(), **owner})
        raise
    finally:
        if child is None or child.poll() is not None:
            if lease is not None:
                (lease / 'by').unlink(); lease.rmdir()
            if registry is not None:
                (registry / 'cesfl_owner.json').unlink(); registry.rmdir()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['smoke', 'train'], required=True)
    main(p.parse_args().stage)
