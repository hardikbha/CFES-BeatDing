#!/usr/bin/env python3
"""Shared additional72GPU-hour ledger. Accounting only; never allocates/kills jobs."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import time


@contextlib.contextmanager
def locked(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with Path(str(path)+'.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        state=json.loads(path.read_text()) if path.exists() else {
            'authorization':'User: Allow up to 72 additional GPU-hours; maximum three concurrent GPUs',
            'cap_gpu_hours':72.,'max_concurrent_gpus':3,'jobs':{}}
        yield state
        tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,path)


def summary(s,now=None):
    now=time.time() if now is None else now
    used=sum(j['charged_gpu_hours'] for j in s['jobs'].values() if j['state']=='FINISHED')
    active=[j for j in s['jobs'].values() if j['state']=='ACTIVE']
    held=sum(max(j['reserved_gpu_hours'],(now-j['start_unix'])*j['gpus']/3600) for j in active)
    return {'charged':used,'reserved':held,
            'active_gpus':sum(j['gpus'] for j in active),
            'uncommitted':s['cap_gpu_hours']-used-held}


def claim(path,name,gpus,hours,allocation,now=None):
    now=time.time() if now is None else now
    if gpus not in (1,2,3) or not 0<hours<=72 or not allocation.strip():raise ValueError('Invalid allocation/budget')
    with locked(path) as s:
        if name in s['jobs']:raise ValueError('Job ID already recorded; preserve history')
        v=summary(s,now)
        if v['active_gpus']+gpus>s['max_concurrent_gpus']:raise ValueError('Three-GPU concurrency cap')
        if hours>v['uncommitted']+1e-10:raise ValueError('72 additional GPU-hour cap')
        s['jobs'][name]={'state':'ACTIVE','gpus':gpus,'reserved_gpu_hours':hours,'start_unix':now,
                         'allocation_evidence':allocation,'charged_gpu_hours':None}
        return summary(s,now)


def finish(path,name,now=None):
    now=time.time() if now is None else now
    with locked(path) as s:
        j=s['jobs'][name]
        if j['state']!='ACTIVE' or now<j['start_unix']:raise ValueError('Invalid finish')
        j.update(state='FINISHED',end_unix=now,charged_gpu_hours=(now-j['start_unix'])*j['gpus']/3600)
        j['reservation_overrun']=j['charged_gpu_hours']>j['reserved_gpu_hours']
        return summary(s,now)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ledger',required=True)
    sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('claim');q.add_argument('--id',required=True);q.add_argument('--gpus',type=int,required=True)
    q.add_argument('--hours',type=float,required=True,help='Total GPU-hours, not wall hours')
    q.add_argument('--allocation',required=True,help='Actual job ID / verified pool reservation; not inferred ownership')
    q=sub.add_parser('finish');q.add_argument('--id',required=True)
    sub.add_parser('status');a=p.parse_args()
    if a.action=='claim':result=claim(a.ledger,a.id,a.gpus,a.hours,a.allocation)
    elif a.action=='finish':result=finish(a.ledger,a.id)
    else:
        with locked(a.ledger) as state:result=summary(state)
    print(json.dumps(result))
