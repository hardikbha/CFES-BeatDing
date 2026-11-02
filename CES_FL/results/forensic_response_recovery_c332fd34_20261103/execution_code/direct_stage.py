"""User-authorized direct SSH, explicit pool-v2 leases, bounded study budget.
No PBS allocation is claimed. Never edits scientific source/config or others' jobs.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-forensic-response-c332fd34')
SCRIPT=SOURCE/'CES_FL/src/ces_fl_forensic_response.py'
SHA='89530025a00b9b0405fba7026d3eed5f2749d86620174ad3ffde93a44f8b770b'
PY='/home/sachin.chaudhary/.conda/envs/aiface/bin/python'


def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


budget=load(ROOT/'allocated_stage.py','fr_budget')
allocator_path=Path('/home/sachin.chaudhary/ces_fl_sources/ding_ff2_public_s17_20261102T130051Z/CES_FL/scripts/run_ding_ff2_ssh.py')
if budget.sha(allocator_path)!='630bc1811f81a4f9021d84c76ab65aff976e1ffcf93e5faa67a7baac4f547a93':
    raise RuntimeError('Audited shared GPU allocator changed')
pool=load(allocator_path,'fr_pool')


def log(event,**kw):
    print(json.dumps(dict(event=event,host=os.uname().nodename,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)),flush=True)


def wait_file(path,limit=7200):
    until=time.monotonic()+limit
    while not path.exists():
        if time.monotonic()>until:
            raise RuntimeError('Dependency timeout: '+str(path))
        failed=list(ROOT.glob('ffpp_checks_*_END.json'))
        if any(json.loads(p.read_text()).get('state')=='FAILED' for p in failed):
            raise RuntimeError('Peer CUDA check failed; do not train')
        time.sleep(10)
    return json.loads(path.read_text())


def reserve(sid,hours):
    with budget.ledger() as d:
        if any(s['id']==sid for s in d['gpu_sessions']):
            raise RuntimeError('Duplicate stage reservation '+sid)
        committed=sum((s['reserved_gpu_hours'] if s['state'] in ('QUEUED','RUNNING') else s.get('elapsed_seconds',0)/3600)
                      for s in d['gpu_sessions'])
        if committed+hours>d['initial_total_gpu_hours']:
            raise RuntimeError('Initial24TOTAL GPU-hour budget would be exceeded')
        d['gpu_sessions'].append(dict(id=sid,ngpus=1,reserved_gpu_hours=hours,elapsed_seconds=0.,state='QUEUED',
                                     route='user_authorized_direct_ssh_pool_v2',pbs_allocation_claimed=False))


def claim(sid):
    pool.LOCKS.mkdir(exist_ok=True)
    until=time.monotonic()+7200
    while time.monotonic()<until:
        with pool.mutex():
            candidates,_=pool.snapshot(16000)
            utilization={int(r.split(',')[0]):int(r.split(',')[1]) for r in subprocess.check_output(
                ['nvidia-smi','--query-gpu=index,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()}
            candidates.sort(key=lambda c:(utilization[c['index']]>10,-c['available_mib']))
            if candidates:
                c=candidates[0];directory=pool.LOCKS/f"gpu{c['index']}"
                if not directory.exists():
                    directory.mkdir();pool.write(directory/'pool_owner.json',dict(shared=True,v=2,pid=0,wid='pool_v2_shared'))
                entry=directory/f'task_{sid}.json'
                if entry.exists():raise RuntimeError('Duplicate GPU lease')
                pool.write(entry,dict(name=sid,wid=sid,pid=os.getpid(),pgid=None,cls='reader',need=16000,time=pool.stamp()))
                c['utilization_at_claim']=utilization[c['index']]
                return c,entry
        log('WAITING_FOR_SAFE_GPU_LEASE',stage=sid)
        time.sleep(30)
    raise RuntimeError('No compatible GPU lease within2hours; existing jobs preserved')


def stage(sid,hours,commands):
    reserve(sid,hours);card,entry=claim(sid);started=time.monotonic();stop=threading.Event();child=None;status='FAILED'
    with budget.ledger() as data:
        item=next(s for s in data['gpu_sessions'] if s['id']==sid)
        item.update(state='RUNNING',host=os.uname().nodename,controller_pid=os.getpid(),card=card,
                    pbs_allocation_claimed=False)
    def tick():
        while not stop.wait(15):
            with budget.ledger() as d:
                item=next(s for s in d['gpu_sessions'] if s['id']==sid);item['elapsed_seconds']=time.monotonic()-started
    thread=threading.Thread(target=tick,daemon=True);thread.start()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=card['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    bootstrap="import sys,torch,runpy,json,os; assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported(); assert 'H100' in torch.cuda.get_device_name(0); torch.cuda.set_per_process_memory_fraction(.16); print(json.dumps(dict(event='CUDA_BINDING',gpu=torch.cuda.get_device_name(0),visible_uuid=os.environ['CUDA_VISIBLE_DEVICES'],device_count=torch.cuda.device_count(),bf16=torch.cuda.is_bf16_supported(),torch=str(torch.__version__),cuda=torch.version.cuda)),flush=True); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    log('LEASE_ASSIGNED',stage=sid,device=card,pbs_allocation_claimed=False)
    try:
        for args,log_path in commands:
            if budget.sha(SCRIPT)!=SHA:raise RuntimeError('Pinned source changed')
            command=[PY,'-u','-c',bootstrap,str(SCRIPT),*args]
            with Path(log_path).open('x') as stream:
                stream.write(json.dumps(dict(command=command,source_commit='c332fd34fb603b8919b15fee08e8d022fb9f57c8',
                        cuda_visible_devices=card['uuid'],lease=str(entry),resource_allocator_fraction=.16))+'\n');stream.flush()
                child=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                with pool.mutex():
                    e=pool.read(entry);e['pgid']=child.pid;pool.write(entry,e)
                with budget.ledger() as d:
                    item=next(s for s in d['gpu_sessions'] if s['id']==sid);item['child_pid']=child.pid;item['log']=str(log_path)
                if child.wait()!=0:raise RuntimeError('Stage command failed; preserve '+str(log_path))
        status='COMPLETED'
    finally:
        stop.set();thread.join(timeout=20)
        elapsed=time.monotonic()-started
        with budget.ledger() as d:
            item=next(s for s in d['gpu_sessions'] if s['id']==sid);item.update(state=status,elapsed_seconds=elapsed)
        with pool.mutex():
            if entry.exists():
                e=pool.read(entry)
                if e.get('pid')==os.getpid() and not pool.alive(e.get('pgid'),True):entry.unlink()
        budget.write(ROOT/f'{sid}_END.json',dict(state=status,elapsed_seconds=elapsed,gpu_hours=elapsed/3600,
                     ngpus=1,source_sha256=SHA,card=card,pbs_allocation_claimed=False))
        log('LEASE_STAGE_END',stage=sid,state=status,gpu_hours=elapsed/3600)


def readiness(run):
    c=json.loads((run/'config.json').read_text());digest=__import__('hashlib').sha256(json.dumps(c,sort_keys=True,allow_nan=False).encode()).hexdigest()
    pre=json.loads((run/'preflight/PREFLIGHT.json').read_text())
    q=json.loads((run/'qualification.json').read_text())
    if pre['status']!='PASS' or pre['source_sha256']!=SHA or pre['config_digest']!=digest or q['source_sha256']!=SHA or q['config_digest']!=digest:
        raise RuntimeError('Source/config/preflight/operator qualification mismatch')
    return digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--role',choices=['primary','secondary'],required=True)
    p.add_argument('--train',choices=['C','D']);a=p.parse_args();run=Path(a.run);digest=readiness(run)
    import fcntl
    lock=(ROOT/f'{a.role}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args=lambda arm:['dry-run','--config',str(run/'config.json'),'--preflight',str(run/'preflight'),
                    '--arm',arm,'--device','cuda','--out',str(run/f'dry_{arm}')]
    if a.role=='secondary':
        smoke=wait_file(run/'software_cuda/SMOKE_RESULT.json')
        if smoke['status']!='PASS' or smoke['source_sha256']!=SHA or not smoke['cuda_tested']:raise RuntimeError('Fresh CUDA smoke failed')
    commands=[]
    if a.role=='primary':commands.append((['smoke','--device','cuda','--out',str(run/'software_cuda')],run/'software_cuda.log'))
    for arm in (('A','C') if a.role=='primary' else ('B','D')):commands.append((args(arm),run/f'dry_{arm}.log'))
    stage('ffpp_checks_'+a.role,1.,commands)
    if not a.train:return
    for arm in ('A','B','C','D'):
        receipt=wait_file(run/f'dry_{arm}/DRY_RUN.json')
        if receipt['status']!='PASS' or receipt['source_sha256']!=SHA or receipt['config_digest']!=digest or receipt['device']!='cuda':
            raise RuntimeError('Source/config/CUDA-matched real backward check missing')
    readiness(run)
    command=['train','--config',str(run/'config.json'),'--preflight',str(run/'preflight'),'--arm',a.train,'--device','cuda',
             '--software-receipt',str(run/'software_cuda/SMOKE_RESULT.json'),
             '--dry-run-receipt',str(run/f'dry_{a.train}/DRY_RUN.json'),'--qualification',str(run/'qualification.json'),
             '--out',str(run/a.train)]
    stage('ffpp_train_'+a.train,3.5,[(command,run/f'train_{a.train}.log')])
    result='COMPLETED_VERIFIED' if (run/a.train/'DONE').exists() else 'INCOMPLETE_EPOCH_BOUNDARY_STOP'
    log('SCIENTIFIC_ARM_STATE',arm=a.train,state=result)


if __name__=='__main__':main()
