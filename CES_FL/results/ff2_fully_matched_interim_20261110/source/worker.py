"""Finite authorised FF2 job queue. No adaptive tuning or automatic retry."""
import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

SOURCE=Path(__file__).resolve().parent
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_fully_matched_20261110_r1')
spec=importlib.util.spec_from_file_location('ff2_gpu_resources',SOURCE/'gpu_pool.py')
pool=importlib.util.module_from_spec(spec);spec.loader.exec_module(pool)
PYTHON='/home/sachin.chaudhary/.conda/envs/aiface/bin/python'

@contextlib.contextmanager
def queue():
    f=(ROOT/'queue.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX)
    state=pool.read(ROOT/'QUEUE.json')
    try:yield state
    finally:pool.write(ROOT/'QUEUE.json',state);fcntl.flock(f,fcntl.LOCK_UN);f.close()

def available():
    with queue() as data:
        return any(j['state']=='PREPARED' and (j['method']!='ding_sfd_gda' or (ROOT/'SFD_FROZEN.json.sha256.json').exists()) for j in data['jobs'])

def claim_job(worker,card):
    with queue() as data:
        eligible=[j for j in data['jobs'] if j['state']=='PREPARED' and
                  (j['method']!='ding_sfd_gda' or (ROOT/'SFD_FROZEN.json.sha256.json').exists())]
        if not eligible:return None
        # Finish TRAIN-only SFD first; then prioritise the direct seed17 table.
        def rank(j):return (-1 if j['method']=='sfd' else (0 if j['method']=='ding_sfd_gda' else 1),[17,7,27].index(j['seed']),data['jobs'].index(j))
        item=min(eligible,key=rank)
        item.update(state='RUNNING',worker=worker,host=os.uname().nodename,gpu=card,started_unix=time.time())
        return dict(item)

def update_job(name,**kwargs):
    with queue() as data:next(j for j in data['jobs'] if j['name']==name).update(**kwargs)

def run_command(args,log,card,entry,deadline):
    bootstrap=("import sys,os,torch,runpy,json;assert torch.cuda.is_available() and torch.cuda.device_count()==1;"
       "assert 'H100' in torch.cuda.get_device_name(0);torch.cuda.set_per_process_memory_fraction(.18);"
       "print(json.dumps(dict(event='CUDA_BINDING',uuid=os.environ['CUDA_VISIBLE_DEVICES'],device=torch.cuda.get_device_name(0),torch=str(torch.__version__))),flush=True);"
       "sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')")
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=card['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
       OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'runtime_deps')+os.pathsep+env.get('PYTHONPATH',''))
    argv=[PYTHON,'-u','-c',bootstrap,str(SOURCE/'src/shared.py'),*args]
    with Path(log).open('x') as f:
        f.write(json.dumps({'command':argv,'lease':str(entry),'gpu':card})+'\n');f.flush()
        child=subprocess.Popen(argv,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        with pool.mutex():e=pool.read(entry);e['pgid']=child.pid;pool.write(entry,e)
        try:code=child.wait(timeout=max(1,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise RuntimeError('Own matched task exceeded its bounded time allowance')
        if code:raise RuntimeError('Stage failed; preserve '+str(log))

def execute(job,card,entry,worker):
    name=job['name'];root=ROOT/name;started=time.monotonic();deadline=started+6.25*3600
    if job['method']=='sfd':
        root.mkdir(exist_ok=True);binding=pool.read(ROOT/'bce_s17/CONFIG_BINDING.json')
        run_command(['sfd','--config',binding['config_path'],'--preflight',str(ROOT/'bce_s17/preflight'),
                     '--out',str(ROOT/'SFD_FROZEN.json'),'--device','cuda'],root/'sfd.log',card,entry,deadline)
        if pool.read(ROOT/'SFD_FROZEN.json')['status']!='PASS':raise RuntimeError('SFD did not pass')
        update_job(name,state='COMPLETED_VERIFIED',elapsed_seconds=time.monotonic()-started,finished_unix=time.time())
        return
    binding=pool.read(root/'CONFIG_BINDING.json');cfg=binding['config_path']
    if pool.sha(cfg)!=binding['config_sha256']:raise RuntimeError('Frozen configuration changed')
    pool.write(root/'ALLOCATION.json',dict(host=os.uname().nodename,gpu=card,worker=worker,lease_path=str(entry),
        source_commit=pool.read(ROOT/'SOURCE_CHECKOUT.json')['commit'],route='explicit_shared_pool_v2',
        exclusive_gpu_claimed=False,pbs_allocation_claimed=False,started_unix=time.time()))
    run_command(['dry','--config',cfg,'--preflight',str(root/'preflight'),'--qualification',str(root/'qualification.json'),
                 '--out',str(root/'CUDA_BACKWARD.json'),'--device','cuda'],root/'cuda_check.log',card,entry,deadline)
    dry=pool.read(root/'CUDA_BACKWARD.json')
    if dry['status']!='PASS' or not dry['bn_frozen']:raise RuntimeError('CUDA backward did not pass')
    if dry['parameters']!={'total':20809001,'trainable':6790433}:raise RuntimeError('Architecture/trainability differs')
    with queue() as data:
        proofs=data.setdefault('initialisation_and_exposure_by_seed',{})
        observed={k:dry[k] for k in ['initial_parameter_fingerprint','data_forward_fingerprint','parameters']}
        prior=proofs.setdefault(str(job['seed']),observed)
        if prior!=observed:raise RuntimeError('Initial parameters or actual forward inputs differ across methods')
    update_job(name,stage='TRAINING')
    run_command(['train','--config',cfg,'--preflight',str(root/'preflight'),'--qualification',str(root/'qualification.json'),
                 '--out',str(root/'training'),'--device','cuda'],root/'train.log',card,entry,deadline)
    result=pool.read(root/'training/RUN_VERIFIED.json')
    if result['status']!='COMPLETED_VERIFIED' or result['epochs_completed']!=10 or result['test_evaluated']:
        raise RuntimeError('Complete selected-model replay required')
    if pool.sha(root/'training/best.pt')!=result['selected_checkpoint_sha256']:raise RuntimeError('Checkpoint hash differs')
    update_job(name,stage='FROZEN_TEST_EVALUATION')
    run_command(['evaluate','--config',cfg,'--training',str(root/'training'),'--out',str(root/'test'),
                 '--device','cuda'],root/'test.log',card,entry,deadline)
    report=pool.read(root/'test/metrics.json')
    if report['calibrated']['overall']['n']!=87564 or report['test_used_for_selection']:
        raise RuntimeError('Incorrect TEST population or selection policy')
    update_job(name,state='COMPLETED_VERIFIED',stage='DONE',elapsed_seconds=time.monotonic()-started,finished_unix=time.time())

def main():
    p=argparse.ArgumentParser();p.add_argument('--slot',type=int,choices=range(4),required=True);a=p.parse_args()
    worker=f'worker_{os.uname().nodename}_{a.slot}';folder=ROOT/worker;folder.mkdir(exist_ok=True)
    lock=(folder/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=pool.verify_source();tests=pool.read(ROOT/'UNIT_CHECKS.json')
    if tests['status']!='PASS' or tests['log_sha256']!=pool.sha(ROOT/'unit_tests.log'):raise RuntimeError('Current software checks required')
    if pool.read(ROOT/'MATRIX_VERIFIED.json')['status']!='PASS':raise RuntimeError('Common recipe matrix not verified')
    dependency=ROOT/'DEPENDENCIES.json'
    if pool.sha(dependency)!=source['dependencies_sha256']:raise RuntimeError('Dependency manifest changed')
    for relative,expected in pool.read(dependency)['files'].items():
        if pool.sha(ROOT/'runtime_deps'/relative)!=expected:raise RuntimeError('Dependency bytes changed')
    until=time.monotonic()+24*3600
    while time.monotonic()<until:
        with queue() as state:
            remaining=any(j['state'] in ('PREPARED','RUNNING') for j in state['jobs'])
        if not remaining:break
        if not available():time.sleep(15);continue
        card=entry=None
        try:
            card,entry=pool.claim(worker)
            job=claim_job(worker,card)
            if job is None:continue
            try:execute(job,card,entry,worker)
            except Exception as error:
                root=ROOT/job['name'];root.mkdir(exist_ok=True)
                pool.write(root/'FAILURE.json',dict(error=str(error),observed_unix=time.time(),no_automatic_retry=True))
                update_job(job['name'],state='FAILED',error=str(error),finished_unix=time.time())
        except Exception as error:
            pool.write(folder/'RESOURCE_WAIT.json',dict(error=str(error),observed_unix=time.time()))
            time.sleep(15)
        finally:
            if entry is not None:
                with pool.mutex():
                    if entry.exists() and pool.read(entry).get('pid')==os.getpid() and not pool.alive(pool.read(entry).get('pgid'),True):entry.unlink()
    pool.write(folder/'WORKER_END.json',dict(status='FINITE_QUEUE_WORKER_ENDED',observed_unix=time.time()))

if __name__=='__main__':main()
