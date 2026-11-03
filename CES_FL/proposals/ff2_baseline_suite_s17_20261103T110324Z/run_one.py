#!/usr/bin/env python3
"""Explicit one-method worker inside an already verified one-GPU allocation.

No scheduler submission, other-job interaction, TEST inference or model download.
The parent limits only its own subprocesses and accounts its held allocation.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import budget

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',required=True,choices=['bce','dag','daw'])
    p.add_argument('--work-root',required=True);p.add_argument('--sources',required=True)
    p.add_argument('--ledger',required=True);p.add_argument('--allocation-record',required=True)
    p.add_argument('--session-id',required=True);p.add_argument('--hours',type=float,default=18.)
    p.add_argument('--resume',action='store_true');p.add_argument('--profile-only',action='store_true')
    a=p.parse_args();start=time.time()
    if not math.isfinite(a.hours) or not 0.1<=a.hours<=24:p.error('Reserve 0.1–24 one-GPU hours within shared72h')
    if not a.session_id or Path(a.session_id).name!=a.session_id:p.error('Simple unique session ID required')
    run=Path(a.work_root).resolve()/a.method
    sources=Path(a.sources).resolve();ledger=Path(a.ledger).resolve()
    if any(path==ROOT or ROOT in path.parents for path in (run,sources,ledger)):raise ValueError('Keep private runtime files outside source')
    if subprocess.run(['git','-C',str(ROOT),'symbolic-ref','-q','HEAD'],stdout=subprocess.DEVNULL).returncode==0:
        raise ValueError('Use an immutable detached worktree')
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():raise ValueError('Dirty source checkout')
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    allocation=json.loads(Path(a.allocation_record).read_text())
    if allocation.get('gpus')!=1 or not all(allocation.get(k) for k in ('allocation_id','verified_by','evidence','cuda_visible_devices')):
        raise ValueError('Record actual scheduler/pool ownership, verifier and binding first')
    if allocation['cuda_visible_devices']!=os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Allocation/CUDA binding differs')
    if os.environ.get('PBS_JOBID') and allocation['allocation_id']!=os.environ['PBS_JOBID']:raise ValueError('Wrong PBS allocation')
    config=HERE/'configs'/f'{a.method}.json';preflight=Path(a.work_root).resolve()/f'preflight_{a.method}'
    if not (preflight/'PREFLIGHT.json').is_file():raise ValueError('Run CPU input preflight before reserving GPU')
    training=run/'training'
    if (training/'DONE').exists():raise ValueError('Already complete: verify/publish, do not rerun')
    if a.resume and not (training/'last.pt').is_file():raise ValueError('Resume requires last.pt')
    if not a.resume and training.exists():raise ValueError('Existing training: inspect and explicitly resume')
    run.mkdir(parents=True,exist_ok=True)
    with (run/'execution.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        session=run/'sessions'/a.session_id;session.mkdir(parents=True,exist_ok=False)
        budget.claim(ledger,a.session_id,1,a.hours,allocation['allocation_id'],now=start)
        deadline=start+a.hours*3600;child=[None]
        def interrupted(signum,frame):raise KeyboardInterrupt('Worker stop signal')
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
        def command(args,name):
            remaining=deadline-time.time()-120
            if remaining<=0:raise TimeoutError('Allocation budget exhausted')
            write(session/(name+'_command.json'),{'argv':args,'cwd':str(ROOT),'remaining_seconds':remaining})
            with (session/(name+'.log')).open('x') as log:
                child[0]=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                try:code=child[0].wait(timeout=remaining)
                except BaseException:
                    child[0].terminate()
                    try:child[0].wait(timeout=60)
                    except subprocess.TimeoutExpired:child[0].kill();child[0].wait()
                    raise
                finally:child[0]=None
            if code:raise RuntimeError(f'{name} exited{code}; inspect log and preserve failure')
        try:
            write(session/'LAUNCH.json',dict(commit=commit,allocation=allocation,session_id=a.session_id,
                reservation_gpu_hours=a.hours,start_unix=start,method=a.method,resume=a.resume,test_read=False,
                config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()))
            py=sys.executable;src=str(ROOT/'CES_FL/src/ff2_fairness_baselines.py');up=str(sources/'dag_daw')
            command([py,'-c','import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0)); print(torch.__version__)'],'cuda_binding')
            smoke=session/'software_fixture'
            command([py,str(ROOT/'CES_FL/tests/test_ff2_baseline_suite.py'),'integration','--upstream',up,'--out',str(smoke),'--device','cuda'],'cuda_integration')
            dry=session/'dry_run'
            base=[py,'-u',src,'--config',str(config),'--preflight',str(preflight),'--upstream',up,'--device','cuda']
            command(base+['dry-run','--out',str(dry),'--steps','20'],'real_backward')
            receipt=json.loads((dry/'DRY_RUN.json').read_text())
            elapsed_before=0.;fraction=1.
            if a.resume:
                import torch
                checkpoint=torch.load(training/'last.pt',map_location='cpu',weights_only=True)
                c=checkpoint['config'];total=c['epochs']*math.ceil(c['train_count']/c['batch_size'])
                fraction=max(0.,1-checkpoint['steps']/total);elapsed_before=checkpoint['elapsed_seconds'];del checkpoint
            estimate=1.4*receipt['estimated_training_only_gpu_hours']*fraction
            available=max(0.,(deadline-time.time()-180)/3600)
            write(session/'PROJECTION.json',dict(training_remaining_with_40percent_margin_hours=estimate,
                session_remaining_hours=available,measured_gpu_hours=receipt['estimated_training_only_gpu_hours'],
                note='Estimate, not guaranteed. VAL/checkpoint/IO variability can exhaust budget; no silent epoch reduction.'))
            if a.profile_only:
                write(session/'SESSION_RESULT.json',{'status':'PROFILE_ONLY','training_launched':False});return
            if estimate>available:
                write(session/'SESSION_RESULT.json',{'status':'BUDGET_PENDING','training_launched':False});return
            command(base+['train','--out',str(training),'--receipt',str(dry/'DRY_RUN.json'),
                '--software-receipt',str(smoke/'SMOKE_RESULT.json'),
                '--budget-hours',str(elapsed_before/3600+available)]+(['--resume'] if a.resume else []),'train')
            write(session/'SESSION_RESULT.json',{'status':'COMPLETED_VERIFIED','training_launched':True})
        except BaseException as exc:
            write(session/'SESSION_RESULT.json',dict(status='STOPPED_REQUIRES_INSPECTION',error=str(exc),type=type(exc).__name__))
            raise
        finally:
            # SIGKILL/node loss cannot execute this. Keep ACTIVE claims and reconcile
            # actual scheduler end time explicitly; never erase reserved usage.
            write(session/'BUDGET_FINISH.json',budget.finish(ledger,a.session_id))


if __name__=='__main__':main()
