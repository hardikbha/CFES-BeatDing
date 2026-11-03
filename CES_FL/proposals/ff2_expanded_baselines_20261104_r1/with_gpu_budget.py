#!/usr/bin/env python3
"""Run one explicit command under the EXISTING shared72h FF2 ledger."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('shared_budget', HERE.parent/'ff2_baseline_suite_s17_20261103T110324Z/budget.py')
budget = importlib.util.module_from_spec(spec); spec.loader.exec_module(budget)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger', type=Path, required=True); p.add_argument('--allocation', type=Path, required=True)
    p.add_argument('--session', required=True); p.add_argument('--hours', type=float, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--projection', type=Path)
    p.add_argument('command', nargs=argparse.REMAINDER); a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command or not .01 <= a.hours <= 24: p.error('Command and reservation0.01–24h required')
    if not a.ledger.is_file(): raise ValueError('Existing shared FF2 ledger required; do not create a second72h allowance')
    state = json.loads(a.ledger.read_text())
    if state.get('cap_gpu_hours') != 72 or state.get('max_concurrent_gpus') != 3: raise ValueError('Wrong ledger scope')
    alloc = json.loads(a.allocation.read_text())
    if alloc.get('gpus') != 1 or not all(alloc.get(k) for k in ('allocation_id','verified_by','evidence','cuda_visible_devices')):
        raise ValueError('Actual one-GPU allocation evidence required')
    if alloc['cuda_visible_devices'] != os.environ.get('CUDA_VISIBLE_DEVICES'): raise ValueError('Binding mismatch')
    if os.environ.get('PBS_JOBID') and alloc['allocation_id'] != os.environ['PBS_JOBID']: raise ValueError('Wrong PBS job')
    if subprocess.run(['git','-C',str(ROOT),'symbolic-ref','-q','HEAD'],capture_output=True).returncode == 0:
        raise ValueError('Use a detached immutable source checkout')
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip(): raise ValueError('Dirty source')
    if a.projection:
        r = json.loads(a.projection.read_text())
        if r.get('status') != 'PASS' or 1.4*r['estimated_training_only_gpu_hours'] + .25 > a.hours:
            raise ValueError('Full training projection with margin does not fit reservation')
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.time(); child = None; status = 'FAILED'; code = None
    budget.claim(a.ledger, a.session, 1, a.hours, alloc['allocation_id'], now=start)
    def interrupted(*_): raise KeyboardInterrupt('Own allocated command interrupted')
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,interrupted)
    try:
        with (a.out/'command.log').open('x') as log:
            child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code = child.wait(timeout=max(1.,a.hours*3600-60)); status = 'COMPLETED' if code == 0 else 'FAILED'
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
        accounting = budget.finish(a.ledger,a.session)
        receipt = dict(status=status,exit_code=code,argv=command,allocation=alloc,start_unix=start,end_unix=time.time(),
                       accounting=accounting,scope='wrapper-held time; operator must reconcile scheduler startup/cleanup bookends')
        (a.out/'SESSION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if code: raise SystemExit(code)


if __name__ == '__main__': main()
