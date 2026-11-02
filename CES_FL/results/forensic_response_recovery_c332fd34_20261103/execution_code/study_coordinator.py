"""Finite authorized A-D execution coordinator. No TEST or fabricated review.
Scientific source/configuration remain pinned; only documented resource resumes.
"""
import datetime
import fcntl
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fr_followups',ROOT/'followup_stage.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
direct=worker.direct
DEADLINE=time.monotonic()+48*3600


def state(phase,**kw):
    obj=dict(observed_management_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             phase=phase,source_commit='c332fd34fb603b8919b15fee08e8d022fb9f57c8',
             source_sha256=direct.SHA,pbs_allocation_claimed=False,test_evaluation=False,**kw)
    direct.budget.write(ROOT/'COORDINATOR_STATUS.json',obj)
    print(json.dumps(obj),flush=True)


def sleep():
    if time.monotonic()>DEADLINE:raise RuntimeError('Finite48hour coordinator deadline; preserve all runs')
    time.sleep(30)


def end_for(sid):
    p=ROOT/(sid+'_END.json')
    return json.loads(p.read_text()) if p.exists() else None


def spawn(host,label,args):
    pidfile=ROOT/(label+'_remote.pid')
    if pidfile.exists():return
    command=['/home/sachin.chaudhary/.conda/envs/aiface/bin/python','-u',str(ROOT/'followup_stage.py'),*args]
    code=('import subprocess,json; from pathlib import Path; '
          f'r=Path({str(ROOT)!r}); p=r/{pidfile.name!r}; '
          'assert not p.exists(),"Duplicate launch record"; '
          f'f=(r/{(label+'_remote.log')!r}).open("x"); '
          f'child=subprocess.Popen({command!r},cwd=r,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); '
          'p.write_text(str(child.pid)+"\\n"); '
          f'(r/{(label+'_launch.json')!r}).write_text(json.dumps(dict(host={host!r},pid=child.pid,command={command!r},pbs_allocation_claimed=False),indent=2)+"\\n"); '
          'print(child.pid)')
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3 -c '+shlex.quote(code)],
                          capture_output=True,text=True,timeout=45)
    if result.returncode:raise RuntimeError('SSH launch failed: '+result.stderr)
    state('FOLLOWUP_LAUNCHED',host=host,label=label,remote_pid=result.stdout.strip(),command=command)


def ensure(dataset,arm,host,initial_sid):
    run=worker.root_for(dataset);sid=initial_sid;allowance=2.
    while True:
        if worker.verified(run,arm):return
        e=end_for(sid)
        if not e:
            state('WAITING_ARM',dataset=dataset,arm=arm,stage=sid)
            sleep();continue
        if e['state']!='COMPLETED':raise RuntimeError('Failed stage preserved: '+sid)
        status_path=run/arm/'STATUS.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('status')!='INCOMPLETE_EPOCH_BOUNDARY_STOP' or not (run/arm/'last.pt').exists():
            raise RuntimeError('No verified completion or documented resumable boundary: '+str(run/arm))
        allowance+=1
        if allowance>4:raise RuntimeError('More than4 cumulative training hours required; inspect before more compute')
        label=f'{dataset}_{arm}_resume{allowance}'
        sid=f'{dataset}_{arm}_resume{allowance}'
        state('RESUME_PLANNED',dataset=dataset,arm=arm,completed_epochs=status['completed_epochs'],
              planned_epochs=10,cumulative_training_allowance_hours=allowance,
              same_source_config_qualification_output=True,no_test_tuning=True)
        spawn(host,label,['--dataset',dataset,'--mode','arm','--arm',arm,'--resume','--budget-hours',str(allowance)])


def pair(dataset,arms,initial=False):
    # Start both controls/paired arms before waiting; each obtains its own lease.
    if not initial:
        for arm,host in zip(arms,('gpu01','gpu02')):
            if not worker.verified(worker.root_for(dataset),arm):
                spawn(host,f'{dataset}_{arm}_initial',['--dataset',dataset,'--mode','arm','--arm',arm])
    for arm,host in zip(arms,('gpu01','gpu02')):
        ensure(dataset,arm,host,'ffpp_train_'+arm if initial else f'{dataset}_{arm}_initial')


def ff2_prepare():
    run=worker.root_for('ff2');pre=run/'preflight/PREFLIGHT.json'
    while not pre.exists():
        log=run/'preflight.log'
        if log.exists() and 'Traceback (most recent call last)' in log.read_text():
            raise RuntimeError('FairForge-2 preflight failed; preserve it and block dependent stages')
        state('WAITING_FF2_CPU_PREFLIGHT');sleep()
    worker.config(run)
    preview=run/'operator_preview/preview_PRIVATE.json'
    if not preview.exists():
        out=run/'operator_preview'
        if out.exists() and any(out.iterdir()):raise RuntimeError('Incomplete preview exists; inspect instead of overwriting')
        with (run/'preview.log').open('x') as stream:
            subprocess.run(['/home/sachin.chaudhary/.conda/envs/aiface/bin/python','-u',str(direct.SCRIPT),'preview',
                            '--config',str(run/'config.json'),'--preflight',str(run/'preflight'),'--out',str(out)],
                           cwd=direct.SOURCE,stdout=stream,stderr=subprocess.STDOUT,check=True)
    # Never writes review/qualification. A real reviewer must supply them.
    state('FF2_PREVIEW_READY_NAMED_REVIEW_REQUIRED',preview_directory=str(run/'operator_preview'))


def run_checks(dataset):
    run=worker.root_for(dataset)
    paths=[run/'software_cuda/SMOKE_RESULT.json']+[run/f'dry_{a}/DRY_RUN.json' for a in 'ABCD']
    if all(p.exists() for p in paths):
        _,digest=worker.config(run)
        for p in paths:
            r=json.loads(p.read_text())
            if r['status']!='PASS' or r['source_sha256']!=direct.SHA or r['device']!='cuda':
                raise RuntimeError('Incompatible existing CUDA receipt: '+str(p))
            if 'config_digest' in r and r['config_digest']!=digest:raise RuntimeError('Backward config mismatch')
        return
    sid=dataset+'_checks_all'
    spawn('gpu01',sid,['--dataset',dataset,'--mode','checks'])
    while not end_for(sid):state('WAITING_CUDA_CHECKS',dataset=dataset);sleep()
    if end_for(sid)['state']!='COMPLETED':raise RuntimeError('Fresh CUDA/real-input check failed: '+dataset)
    run_checks(dataset)


def main():
    lock=(ROOT/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        state('FFPP_C_D_RUNNING')
        pair('ffpp','CD',initial=True)
        state('FFPP_C_D_VERIFIED_START_A_B')
        pair('ffpp','AB')
        state('FFPP_ALL_FOUR_VERIFIED')
        ff2_prepare();run_checks('ff2')
        run=worker.root_for('ff2')
        if not (run/'qualification.json').exists():
            # Original-only controls are valid while intervention review is pending.
            state('FF2_C_D_BLOCKED_NAMED_REVIEW_START_ELIGIBLE_A_B')
            pair('ff2','AB')
            while not (run/'qualification.json').exists():state('FF2_C_D_BLOCKED_NAMED_REVIEW');sleep()
        direct.readiness(run)
        pair('ff2','CD')
        if not all(worker.verified(run,a) for a in 'AB'):pair('ff2','AB')
        state('ALL_EIGHT_ARMS_COMPLETED_VERIFIED')
    except BaseException as error:
        state('STOPPED_PRESERVE_AND_INSPECT',error=repr(error))
        raise


if __name__=='__main__':main()
