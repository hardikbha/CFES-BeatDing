"""Extra finite queue consumers; all training executes from the pinned FF2 source."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

SOURCE = Path('/home/sachin.chaudhary/ces_fl_sources/ff2-fully-matched-20261110')
CONTROL = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
import worker as study

pool = study.pool
ROOT = study.ROOT


def verify():
    binding = pool.read(ROOT / 'CAPACITY_CONTROL_BINDING.json')
    if (CONTROL / '.git/HEAD').read_text().strip() != binding['control_commit']:
        raise RuntimeError('Detached pinned capacity controller required')
    if pool.sha(__file__) != binding['controller_sha256']:
        raise RuntimeError('Capacity controller bytes changed')
    source = pool.verify_source()
    if source['commit'] != binding['training_source_commit']:
        raise RuntimeError('Training source differs from capacity binding')
    tests = pool.read(ROOT / 'UNIT_CHECKS.json')
    if tests['status'] != 'PASS' or tests['log_sha256'] != pool.sha(ROOT / 'unit_tests.log'):
        raise RuntimeError('Current training software checks required')
    if pool.read(ROOT / 'MATRIX_VERIFIED.json')['status'] != 'PASS':
        raise RuntimeError('Common recipe matrix not verified')
    dependency = ROOT / 'DEPENDENCIES.json'
    if pool.sha(dependency) != source['dependencies_sha256']:
        raise RuntimeError('Dependency manifest changed')
    for relative, expected in pool.read(dependency)['files'].items():
        if pool.sha(ROOT / 'runtime_deps' / relative) != expected:
            raise RuntimeError('Dependency bytes changed')
    return binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slot', type=int, choices=range(4, 8))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    binding = verify()
    if args.verify_only:
        print(json.dumps(dict(status='PASS', **binding)), flush=True)
        return
    if args.slot is None:
        parser.error('--slot is required for execution')
    name = f'worker_{os.uname().nodename}_{args.slot}'
    folder = ROOT / name
    folder.mkdir(exist_ok=True)
    lock = (folder / 'execution.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pool.write(folder / 'CAPACITY_WORKER.json', dict(
        status='STANDBY', pid=os.getpid(), host=os.uname().nodename,
        slot=args.slot, started_unix=time.time(), **binding))
    until = time.monotonic() + 24 * 3600
    while time.monotonic() < until and study.available():
        entry = None
        try:
            # The original pool atomically checks memory, utilization and shared
            # leases. It excludes every GPU already running another FF2 arm.
            card, entry = pool.claim(name)
            job = study.claim_job(name, card)
            if job is None:
                continue
            pool.write(folder / 'CAPACITY_WORKER.json', dict(
                status='EXECUTING', pid=os.getpid(), job=job['name'],
                host=os.uname().nodename, gpu=card, observed_unix=time.time(), **binding))
            try:
                study.execute(job, card, entry, name)
            except Exception as error:
                root = ROOT / job['name']
                root.mkdir(exist_ok=True)
                pool.write(root / 'FAILURE.json', dict(
                    error=str(error), observed_unix=time.time(), no_automatic_retry=True))
                study.update_job(job['name'], state='FAILED', error=str(error), finished_unix=time.time())
        except Exception as error:
            pool.write(folder / 'RESOURCE_WAIT.json', dict(
                error=str(error), observed_unix=time.time()))
            time.sleep(15)
        finally:
            if entry is not None:
                with pool.mutex():
                    if (entry.exists() and pool.read(entry).get('pid') == os.getpid()
                            and not pool.alive(pool.read(entry).get('pgid'), True)):
                        entry.unlink()
    pool.write(folder / 'WORKER_END.json', dict(
        status='FINITE_QUEUE_WORKER_ENDED', observed_unix=time.time()))


if __name__ == '__main__':
    main()
