"""Inside one ngpus=1 PBS allocation, lease an actually empty device and run."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

LOCKS = Path('/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA/work/gpulocks')


def write(path, data):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n'); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--protocol', required=True); p.add_argument('--phase', required=True)
    args = p.parse_args(); protocol = json.loads(Path(args.protocol).read_text())
    root = Path(protocol['run_root']); phase = args.phase; host = socket.gethostname().split('.')[0]
    guard = (root/'runtime'/f'{phase}.stage_lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    job = os.environ.get('PBS_JOBID'); assert job, 'A PBS allocation with explicit ngpus is required'
    allocation = subprocess.check_output(['qstat', '-f', job], text=True)
    assert 'ngpus=1' in allocation or 'Resource_List.ngpus = 1' in allocation
    assert not (root/'runtime'/f'{phase}.result.json').exists(), 'Completed phase must not rerun'
    owner = protocol['request_id']+':'+phase+':'+job
    LOCKS.mkdir(parents=True, exist_ok=True)
    lock = None; child = None
    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        started = time.time()
        while lock is None:
            if time.time()-started > 7200:
                raise RuntimeError('No actually empty GPU within2hours of PBS allocation; other jobs left untouched')
            occupied = set(subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines())
            rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            allowed = set(visible.split(',')) if visible else None
            for row in rows:
                index, uuid, free, util = [x.strip() for x in row.split(',')]
                if allowed is not None and index not in allowed and uuid not in allowed:
                    continue
                if uuid in occupied or int(free) < 24576 or int(util) > 10:
                    continue
                candidate = LOCKS/(host+'_'+index)
                try: candidate.mkdir()
                except FileExistsError: continue
                (candidate/'by').write_text(owner+'\n')
                lock = candidate
                # Check again under the shared project lease. Never evict an app.
                now = set(subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines())
                if uuid in now:
                    (candidate/'by').unlink(); candidate.rmdir(); lock = None; continue
                break
            if lock is None:
                state = {'phase': 'WAITING_FOR_EMPTY_GPU', 'stage': phase, 'pbs_job_id': job, 'host': host, 'time': time.time()}
                write(root/'runtime'/f'{phase}.progress.json', state)
                print(json.dumps(state), flush=True); time.sleep(20)
        env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        command = [sys.executable, '-u', str(Path(__file__).with_name('worker.py')), '--protocol', args.protocol, '--phase', phase]
        child = subprocess.Popen(command, env=env, start_new_session=True)
        write(root/'runtime'/f'{phase}.job.json', {'state': 'RUNNING', 'pbs_job_id': job, 'gpu_uuid': uuid, 'host': host,
              'pid': child.pid, 'command': command, 'binding': 'PBS ngpus1 plus an unoccupied physical device and shared project lease'})
        code = child.wait()
        write(root/'runtime'/f'{phase}.job.json', {'state': 'DONE' if code == 0 else 'FAILED', 'exit_code': code,
              'pbs_job_id': job, 'gpu_uuid': uuid, 'host': host})
        if code: raise SystemExit(code)
    except BaseException as e:
        write(root/'runtime'/f'{phase}.failure.json', {'state': 'FAILED', 'error': repr(e), 'pbs_job_id': job, 'time': time.time()})
        raise
    finally:
        if lock is not None and (lock/'by').is_file() and (lock/'by').read_text().strip() == owner:
            (lock/'by').unlink(); lock.rmdir()


if __name__ == '__main__':
    main()
