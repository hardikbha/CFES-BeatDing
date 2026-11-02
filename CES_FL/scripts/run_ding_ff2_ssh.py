"""Finite direct-SSH two-GPU operator. Joins existing shared pool v2 safely.

Never preempts, deletes another task/lease, submits PBS, downloads, or evaluates
TEST. Need/free/headroom are checked together under the pool host mutex.
"""
import argparse
import contextlib
import csv
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

LOCKS=Path('/tmp/scf_gpu_locks')


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def read(path):
    return json.loads(Path(path).read_text())


def write(path,obj):
    path=Path(path)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n')
    os.replace(temp,path)


def alive(pid,group=False):
    if not isinstance(pid,int) or pid<=0:
        return False
    try:
        (os.killpg if group else os.kill)(pid,0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextlib.contextmanager
def mutex():
    p=LOCKS/'.pool_mutex'
    until=time.time()+30
    while True:
        try:
            p.mkdir()
            break
        except FileExistsError:
            if time.time()>until:
                raise RuntimeError('Pool mutex busy; no stale-lock removal authorized')
            time.sleep(.2)
    try:
        yield
    finally:
        p.rmdir()


def snapshot(need):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used','--format=csv,noheader,nounits'],text=True)
    cards={int(r[0]):dict(index=int(r[0]),uuid=r[1].strip(),free=int(r[2]),used=int(r[3])) for r in csv.reader(text.splitlines())}
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    apps=[]
    for r in csv.reader(text.splitlines()):
        if len(r)!=3:
            continue
        try:
            apps.append((r[0].strip(),os.getpgid(int(r[1])),int(r[2])))
        except (ValueError,ProcessLookupError,PermissionError):
            pass
    candidates=[]
    for i,c in cards.items():
        d=LOCKS/f'gpu{i}'
        if d.exists():
            try:
                owner=read(d/'pool_owner.json')
                if not owner.get('shared'):
                    continue
                entries=[read(f) for f in d.glob('task_*.json')]
                entries=[e for e in entries if alive(e.get('pid')) or alive(e.get('pgid'),True)]
            except (OSError,ValueError,TypeError):
                continue
        else:
            entries=[]
        reserve=0
        for e in entries:
            actual=sum(m for uuid,pgid,m in apps if uuid==c['uuid'] and pgid==e.get('pgid'))
            reserve+=max(0,int(e.get('need',0))-actual)
        c.update(reserved_not_yet_used_mib=reserve,available_mib=c['free']-reserve)
        if c['available_mib']>=need+4096:
            candidates.append(c)
    return sorted(candidates,key=lambda c:c['available_mib'],reverse=True),cards


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--python',required=True)
    p.add_argument('--preflight',default='preflight')
    p.add_argument('--need-mib',type=int,default=20000)
    p.add_argument('--wait-hours',type=float,default=12)
    a=p.parse_args()
    source=Path(a.source).resolve()
    run=Path(a.run).resolve()
    cfg=Path(a.config).resolve()
    lock=(run/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (run/'training').exists() or (run/'cuda_smoke').exists():
        raise RuntimeError('Prior execution output exists; do not duplicate or overwrite')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    branch=subprocess.run(['git','symbolic-ref','-q','HEAD'],cwd=source,capture_output=True)
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
    if branch.returncode==0 or dirty:
        raise RuntimeError('Execution source must be detached and clean')
    pre=read(run/a.preflight/'PREFLIGHT.json')
    if pre['status']!='PASS' or pre['config_sha256']!=hashlib.sha256(cfg.read_bytes()).hexdigest():
        raise RuntimeError('Missing or stale input preflight')
    for name,digest in pre['source_hashes'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Source changed since preflight')
    cpu=read(run/'CPU_TESTS_PASS.json')
    if cpu['status']!='PASS' or cpu['source_hashes']!=pre['source_hashes']:
        raise RuntimeError('CPU checks missing')
    available_kib=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
    if available_kib < 128*1024**2:
        raise RuntimeError('Insufficient host RAM headroom; preserve other jobs')
    name=run.name
    status=dict(utc=stamp(),state='WAITING_FOR_TWO_GPUS',commit=commit,host=socket.gethostname(),
                controller_pid=os.getpid(),route='direct_ssh_shared_pool_v2',pbs_allocation_claimed=False,
                requested_cpus=16,exclusive_cpu_reservation=False,need_mib_each=a.need_mib)
    write(run/'STATUS.json',status)
    print(json.dumps(status),flush=True)
    LOCKS.mkdir(exist_ok=True)
    selected=[]
    until=time.time()+3600*a.wait_hours
    while time.time()<until:
        with mutex():
            candidates,cards=snapshot(a.need_mib)
            if len(candidates)>=2:
                # Claim both atomically. Respect every existing shared/exclusive lease.
                chosen=candidates[:2]
                for c in chosen:
                    d=LOCKS/f"gpu{c['index']}"
                    if not d.exists():
                        d.mkdir()
                        write(d/'pool_owner.json',dict(shared=True,v=2,pid=0,wid='pool_v2_shared',time=stamp()))
                    entry=d/f'task_{name}.json'
                    if entry.exists():
                        raise RuntimeError('Existing own-name registry entry; inspect instead of replacing')
                    write(entry,dict(name=name,wid=name,pid=os.getpid(),pgid=None,cls='light',need=a.need_mib,time=stamp()))
                    selected.append((c,entry))
                break
        print(json.dumps(dict(utc=stamp(),state='WAITING_FOR_TWO_GPUS',eligible=len(candidates),cards=cards)),flush=True)
        time.sleep(30)
    if len(selected)!=2:
        raise RuntimeError('No two compatible shared GPUs within wait limit; no training launched')
    try:
        status.update(utc=stamp(),state='CUDA_SMOKE',devices=[c for c,_ in selected])
        write(run/'STATUS.json',status)
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=','.join(c['uuid'] for c,_ in selected),OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',
                   TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        script=source/'CES_FL/src/ding_ff2_public.py'
        for command,outname in [('cuda-smoke','cuda_smoke'),('train','training')]:
            if command=='train':
                receipt=read(run/'cuda_smoke/CUDA_SMOKE_PASS.json')
                if receipt['status']!='PASS' or receipt['source_hashes']!=pre['source_hashes']:
                    raise RuntimeError('CUDA smoke failed or source mismatch: stop launch')
            cmd=[a.python,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=2',
                 str(script),command,'--config',str(cfg),'--out',str(run/outname)]
            logfile=run/'logs'/('cuda_smoke.log' if command=='cuda-smoke' else 'training.log')
            with logfile.open('a') as stream:
                stream.write(json.dumps(dict(utc=stamp(),command=cmd,devices=env['CUDA_VISIBLE_DEVICES'],commit=commit))+'\n')
                stream.flush()
                child=subprocess.Popen(cmd,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                with mutex():
                    for c,entry in selected:
                        e=read(entry);e['pgid']=child.pid;e['time']=stamp();write(entry,e)
                status.update(utc=stamp(),state='CUDA_SMOKE' if command=='cuda-smoke' else 'RUNNING',
                              torchrun_pid=child.pid,pgid=child.pid,command=cmd,log=str(logfile))
                write(run/'STATUS.json',status)
                code=child.wait()
            if code:
                raise RuntimeError(f'{command} failed with exit {code}; see {logfile}')
        status.update(utc=stamp(),state='COMPLETED_VERIFIED')
        write(run/'STATUS.json',status)
    except BaseException as error:
        status.update(utc=stamp(),state='FAILED',error=str(error))
        write(run/'STATUS.json',status)
        raise
    finally:
        # Never terminate another job. Release only our registry entries after our children end.
        with mutex():
            for c,entry in selected:
                if entry.exists():
                    e=read(entry)
                    if e.get('pid')==os.getpid() and not alive(e.get('pgid'),True):
                        entry.unlink()


if __name__=='__main__':
    main()
