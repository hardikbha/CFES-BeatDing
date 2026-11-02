"""Execution glue only: reserved GPU, immutable source, and shared study ledger.
Does not edit inputs, scientific configuration, reviews or model source.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

SOURCE=Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-forensic-response-c332fd34')
SCRIPT=SOURCE/'CES_FL/src/ces_fl_forensic_response.py'
SHA='89530025a00b9b0405fba7026d3eed5f2749d86620174ad3ffde93a44f8b770b'
AUDIT=Path(__file__).resolve().parent
LEDGER=AUDIT/'budget_ledger.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,obj):
    temp=Path(str(path)+'.tmp.'+str(os.getpid()))
    temp.write_text(json.dumps(obj,indent=2)+'\n')
    os.replace(temp,path)


@contextlib.contextmanager
def ledger():
    with (AUDIT/'budget.lock').open('a') as lock:
        until=time.monotonic()+30
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic()>until:
                    raise RuntimeError('Study budget ledger busy; no launch')
                time.sleep(.1)
        data=json.loads(LEDGER.read_text())
        yield data
        data['consumed_gpu_hours']=sum(s.get('elapsed_seconds',0)*s['ngpus']/3600 for s in data['gpu_sessions'])
        data['reserved_gpu_hours']=sum(max(0,s['reserved_gpu_hours']-s.get('elapsed_seconds',0)*s['ngpus']/3600)
                                        for s in data['gpu_sessions'] if s['state'] in ('QUEUED','RUNNING'))
        write(LEDGER,data)


def run(command,log):
    with Path(log).open('x') as stream:
        stream.write(json.dumps(dict(command=command))+'\n');stream.flush()
        subprocess.run(command,cwd=SOURCE,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['checks','train'],required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--session',required=True)
    p.add_argument('--arm',choices=['A','B','C','D'])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--budget-hours',type=float)
    a=p.parse_args();root=Path(a.run)
    job=os.environ.get('PBS_JOBID')
    if not job:
        raise RuntimeError('Requires an actual PBS GPU allocation; no SSH fallback')
    started=time.monotonic();stop=threading.Event()
    with ledger() as data:
        entry=next(s for s in data['gpu_sessions'] if s['id']==a.session)
        if entry['state']!='QUEUED' or entry['ngpus']!=1:
            raise RuntimeError('Missing unique one-GPU budget reservation')
        entry.update(state='RUNNING',pbs_job_id=job,host=os.uname().nodename,pid=os.getpid(),elapsed_seconds=0.)
    def heartbeat():
        while not stop.wait(20):
            with ledger() as data:
                entry=next(s for s in data['gpu_sessions'] if s['id']==a.session)
                entry['elapsed_seconds']=time.monotonic()-started
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    status='FAILED'
    try:
        if sha(SCRIPT)!=SHA:
            raise RuntimeError('Pinned source changed')
        q=subprocess.check_output(['/opt/pbs/bin/qstat','-f',job],text=True,timeout=30)
        (AUDIT/f'{a.session}_qstat_PRIVATE.txt').write_text(q)
        if 'Resource_List.ngpus = 1' not in q:
            raise RuntimeError('PBS does not confirm ngpus=1')
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','').strip()
        if not visible or len(visible.split(','))!=1:
            raise RuntimeError('Scheduler CUDA_VISIBLE_DEVICES is absent/not one GPU; binding unverified, STOP')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Allocated CUDA/bfloat16 binding failed')
        if 'H100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Allocated GPU is not the requested H100')
        free,total=torch.cuda.mem_get_info()
        if free<16*1024**3:
            raise RuntimeError('Allocated GPU has less than16GiB free; no device substitution or batch reduction')
        config=root/'config.json';pre=json.loads((root/'preflight/PREFLIGHT.json').read_text());c=json.loads(config.read_text())
        digest=hashlib.sha256(json.dumps(c,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if pre['status']!='PASS' or pre['source_sha256']!=SHA or pre['config_digest']!=digest:
            raise RuntimeError('Source/config-matched CPU preflight missing')
        write(AUDIT/f'{a.session}_DEVICE.json',dict(pbs_job_id=job,source_sha256=SHA,host=os.uname().nodename,
              cuda_visible_devices=visible,device=torch.cuda.get_device_name(0),torch=str(torch.__version__),
              cuda=torch.version.cuda,bf16=torch.cuda.is_bf16_supported(),free_cuda_bytes=free,total_cuda_bytes=total,
              config_digest=digest))
        base=[sys.executable,'-u',str(SCRIPT)]
        if a.mode=='checks':
            run(base+['smoke','--device','cuda','--out',str(root/'software_cuda')],root/'software_cuda.log')
            result=json.loads((root/'software_cuda/SMOKE_RESULT.json').read_text())
            if result['status']!='PASS' or result['source_sha256']!=SHA or not result['cuda_tested']:
                raise RuntimeError('Fresh CUDA smoke did not pass')
            for arm in ('A','B','C','D'):
                run(base+['dry-run','--config',str(config),'--preflight',str(root/'preflight'),'--arm',arm,
                          '--device','cuda','--out',str(root/f'dry_{arm}')],root/f'dry_{arm}.log')
                result=json.loads((root/f'dry_{arm}/DRY_RUN.json').read_text())
                if result['status']!='PASS' or result['source_sha256']!=SHA or result['config_digest']!=digest:
                    raise RuntimeError(f'Real-input {arm} check failed or mismatched')
            status='CHECKS_PASS'
        else:
            if not a.arm:
                raise RuntimeError('Explicit training arm required')
            command=base+['train','--config',str(config),'--preflight',str(root/'preflight'),
                    '--arm',a.arm,'--device','cuda','--software-receipt',str(root/'software_cuda/SMOKE_RESULT.json'),
                    '--dry-run-receipt',str(root/f'dry_{a.arm}/DRY_RUN.json'),'--out',str(root/a.arm)]
            if a.arm in ('C','D'):
                command+=['--qualification',str(root/'qualification.json')]
            if a.resume:
                command+=['--resume']
            if a.budget_hours is not None:
                command+=['--budget-hours',str(a.budget_hours)]
            run(command,root/f'train_{a.arm}_{a.session}.log')
            status='COMPLETED_VERIFIED' if (root/a.arm/'DONE').exists() else 'INCOMPLETE'
    except BaseException as error:
        write(AUDIT/f'{a.session}_FAILURE.json',dict(status='FAILED',error=repr(error),source_sha256=SHA,pbs_job_id=job))
        raise
    finally:
        stop.set();thread.join(timeout=25)
        with ledger() as data:
            entry=next(s for s in data['gpu_sessions'] if s['id']==a.session)
            entry.update(state=status,elapsed_seconds=time.monotonic()-started)
        write(AUDIT/f'{a.session}_END.json',dict(status=status,elapsed_seconds=time.monotonic()-started,
              ngpus=1,pbs_job_id=job,source_sha256=SHA))


if __name__=='__main__':
    main()
