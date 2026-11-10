"""Finite four-candidate TRAIN/SELECT queue; charges the existing study ledger."""
import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

SOURCE=Path(__file__).resolve().parent
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_response_tuning_20261110_r1')
PYTHON='/home/sachin.chaudhary/.conda/envs/aiface/bin/python'
BUDGET_HELPER=Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/allocated_stage.py')
PREFIX='ff2_response_tuning_20261110_r1_'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


pool=load(SOURCE/'gpu_pool.py','tuning_pool')


@contextlib.contextmanager
def queue():
    with (ROOT/'queue.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);state=pool.read(ROOT/'QUEUE.json')
        try:yield state
        finally:pool.write(ROOT/'QUEUE.json',state);fcntl.flock(f,fcntl.LOCK_UN)


def update(name,**values):
    with queue() as q:next(j for j in q['jobs'] if j['name']==name).update(**values)


def verify():
    r=pool.read(ROOT/'SOURCE_CHECKOUT.json');assert (SOURCE/'.git/HEAD').read_text().strip()==r['commit']
    for n,h in r['source_files'].items():assert pool.sha(SOURCE/n)==h,n
    tests=pool.read(ROOT/'CPU_CHECKS.json');assert tests['status']=='PASS'
    for n,h in tests['log_hashes'].items():assert pool.sha(ROOT/n)==h,n
    assert pool.read(ROOT/'MATRIX_VERIFIED.json')['status']=='PASS'
    assert pool.sha(BUDGET_HELPER)==r['budget_helper_sha256']
    dep=pool.read(ROOT/'DEPENDENCIES.json')
    for n,h in dep['files'].items():assert pool.sha(ROOT/'runtime_deps'/n)==h,n
    return load(BUDGET_HELPER,'tuning_budget')


def claim(worker):
    until=time.monotonic()+7200
    while time.monotonic()<until:
        with queue() as q:
            if not any(j['state']=='PREPARED' for j in q['jobs']):return None,None,None
        with pool.mutex():
            choices=pool.candidates()
            if choices:
                card=choices[0];directory=pool.LOCKS/('gpu'+str(card['index']))
                if not directory.exists():
                    directory.mkdir();pool.write(directory/'pool_owner.json',dict(shared=True,v=2,pid=0,wid='pool_v2_shared'))
                lease=directory/('task_ff2suite_'+worker+'.json');assert not lease.exists()
                with queue() as q:
                    eligible=[j for j in q['jobs'] if j['state']=='PREPARED']
                    if not eligible:return None,None,None
                    j=eligible[0];j.update(state='RUNNING',stage='CUDA_CHECKS',host=os.uname().nodename,worker=worker,gpu=card,started_unix=time.time())
                    job=dict(j)
                pool.write(lease,dict(name='ff2suite_'+worker,wid='ff2suite_'+worker,pid=os.getpid(),pgid=None,cls='reader',need=16000,time=time.time()))
                return job,card,lease
        pool.write(ROOT/worker/'RESOURCE_WAIT.json',dict(status='WAITING_FOR_ELIGIBLE_SHARED_H100',host=os.uname().nodename,observed_unix=time.time()))
        time.sleep(15)
    raise RuntimeError('No GPU within bounded wait; preserve existing jobs')


def execute(job,card,lease,budget):
    name=job['name'];root=ROOT/name;start=time.monotonic();deadline=start+4.20*3600;stop=threading.Event();child=None;status='FAILED'
    def charge(final=False):
        with budget.ledger() as d:
            item=next(x for x in d['gpu_sessions'] if x['id']==PREFIX+name)
            if item['state'] not in ('QUEUED','RUNNING'):raise RuntimeError('Missing live original-ledger reservation')
            item.update(state=status if final else 'RUNNING',host=os.uname().nodename,controller_pid=os.getpid(),
                elapsed_seconds=time.monotonic()-start,card=card,source_commit=pool.read(ROOT/'SOURCE_CHECKOUT.json')['commit'])
    def heartbeat():
        while not stop.wait(15):charge()
    def watchdog():
        while not stop.wait(.5):
            if time.monotonic()>=deadline:
                if child is not None and child.poll() is None:
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    time.sleep(3)
                    if child.poll() is None:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                return
    def command(stage,out,log):
        nonlocal child
        if time.monotonic()>=deadline:raise RuntimeError('Own held-GPU deadline reached')
        binding=pool.read(root/'CONFIG_BINDING.json');cfg=binding['config_path'];assert pool.sha(cfg)==binding['config_sha256']
        argv=[stage,'--config',cfg,'--preflight',str(root/'preflight'),'--qualification',str(root/'qualification.json'),'--out',str(out),'--device','cuda']
        bootstrap=("import torch,os,sys,runpy,json;assert torch.cuda.is_available() and torch.cuda.device_count()==1;"
            "assert 'H100' in torch.cuda.get_device_name(0) and torch.cuda.is_bf16_supported();"
            "assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==os.environ['CUDA_VISIBLE_DEVICES'].removeprefix('GPU-');"
            "torch.cuda.set_per_process_memory_fraction(.18);print(json.dumps(dict(event='CUDA_BINDING',uuid=os.environ['CUDA_VISIBLE_DEVICES'],device=torch.cuda.get_device_name(0),torch=str(torch.__version__))),flush=True);"
            "sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')")
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=card['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
            PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'runtime_deps'))
        with Path(log).open('x') as f:
            f.write(json.dumps(dict(stage=stage,gpu=card,lease=str(lease),source_commit=pool.read(ROOT/'SOURCE_CHECKOUT.json')['commit']))+'\n');f.flush()
            child=subprocess.Popen([PYTHON,'-u','-c',bootstrap,str(SOURCE/'schedule_tuning.py'),*argv],cwd=SOURCE,env=env,
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            with pool.mutex():e=pool.read(lease);e['pgid']=child.pid;pool.write(lease,e)
            code=child.wait(timeout=max(1,deadline-time.monotonic()))
            if code:raise RuntimeError('Failed '+stage+'; preserve '+str(log))
    tick=guard=None
    try:
        charge();tick=threading.Thread(target=heartbeat,daemon=True);guard=threading.Thread(target=watchdog,daemon=True);tick.start();guard.start()
        pool.write(root/'ALLOCATION.json',dict(host=os.uname().nodename,gpu=card,worker=job['worker'],lease_path=str(lease),
            route='explicit_authorized_shared_pool_v2',pbs_allocation_claimed=False,exclusive_GPU_claimed=False,held_hour_limit=4.25,
            native_training_hour_limit=4,original_ledger=str(budget.LEDGER),reservation_id=PREFIX+name))
        command('dry',root/'cuda_checks',root/'cuda_checks.log');dry=pool.read(root/'cuda_checks/CUDA_BACKWARD.json')
        assert dry['status']=='PASS' and dry['bn_frozen'] and dry['parameters']==dict(total=20809001,trainable=6790433)
        fingerprint={k:dry[k] for k in ('initial_parameter_fingerprint','initial_forward_input_fingerprint','parameters')}
        with queue() as q:
            prior=q.setdefault('common_seed17_initialisation_and_exposure',fingerprint)
            if prior!=fingerprint:raise RuntimeError('Different common initialization or actual forward input')
        update(name,stage='TRAINING');command('train',root/'training',root/'train.log')
        result=pool.read(root/'training/TUNING_VERIFIED.json')
        assert result['status']=='COMPLETED_VERIFIED' and result['epochs_completed']==10 and not result['test_evaluated']
        assert pool.sha(root/'training/best.pt')==result['selected_checkpoint_sha256']
        status='COMPLETED_VERIFIED';update(name,state=status,stage='DONE',selected_epoch=result['selected_epoch'],finished_unix=time.time())
    except BaseException as error:
        pool.write(root/'FAILURE.json',dict(error=str(error),no_automatic_retry=True,observed_unix=time.time()))
        update(name,state='FAILED',stage='STOPPED',error=str(error),finished_unix=time.time())
    finally:
        stop.set()
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if tick is not None:tick.join(timeout=30)
        if guard is not None:guard.join(timeout=5)
        charge(final=True)
        with pool.mutex():
            if lease.exists() and pool.read(lease).get('pid')==os.getpid() and not pool.alive(pool.read(lease).get('pgid'),True):lease.unlink()
        pool.write(root/'GPU_ACCOUNTING.json',dict(state=status,actual_held_GPU_hours=(time.monotonic()-start)/3600,cap_GPU_hours=4.25,TEST_CAL_access=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--slot',type=int,choices=(0,1),required=True);a=p.parse_args();worker=f"response_tune_{os.uname().nodename}_{a.slot}";folder=ROOT/worker;folder.mkdir(exist_ok=True)
    with (folder/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);budget=verify();until=time.monotonic()+12*3600
        pool.write(folder/'WORKER_STARTED.json',dict(pid=os.getpid(),host=os.uname().nodename,slot=a.slot,observed_unix=time.time()))
        while time.monotonic()<until:
            with queue() as q:
                prepared=any(j['state']=='PREPARED' for j in q['jobs']);running=any(j['state']=='RUNNING' for j in q['jobs'])
            if not prepared:
                if not running:break
                time.sleep(15);continue
            job,card,lease=claim(worker)
            if job is not None:execute(job,card,lease,budget)
        pool.write(folder/'WORKER_END.json',dict(observed_unix=time.time(),status='FINITE_QUEUE_ENDED'))
    # This helper reports existing VAL data only and cannot start more jobs.
    subprocess.run(['/usr/bin/python3','-S',str(SOURCE/'summarize_tuning.py')],cwd=SOURCE,check=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(ROOT/'runtime_deps'))
    with (folder/'comparison_attempt.log').open('x') as f:
        subprocess.run([PYTHON,'-u',str(SOURCE/'compare_tuning.py')],cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=7200)


if __name__=='__main__':main()
