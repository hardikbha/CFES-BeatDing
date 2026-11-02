"""Observation/publication only for one explicitly authorized run; finite72h.

Cannot launch/restart/cancel training, generate images, or evaluate TEST.
Publishes aggregate text/metrics from this run on its dedicated branch.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):
            h.update(b)
    return h.hexdigest()


def observe(run):
    state=json.loads((run/'STATUS.json').read_text())
    pid=state.get('torchrun_pid')
    remote='import os,json,time; p='+str(int(pid or 0))+'; print(json.dumps(dict(remote_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),torchrun_proc_exists=os.path.exists("/proc/"+str(p)))))'
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',state['host'],'python3 -c '+shlex.quote(remote)],capture_output=True,text=True,timeout=30)
    process=json.loads(result.stdout) if result.returncode==0 else dict(verification='SSH_FAILED',error=result.stderr.strip())
    events=[]
    for line in (run/'logs/training.log').read_text().splitlines():
        try:
            row=json.loads(line)
            if 'event' in row:
                events.append(row)
        except ValueError:
            pass
    stages={}
    for stage in ('classification','gda_corrected'):
        history=run/'training'/stage/'history.jsonl'
        records=[json.loads(line) for line in history.read_text().splitlines()] if history.exists() else []
        stages[stage]=dict(completed_epochs=len(records),latest=records[-1] if records else None)
        best=run/'training'/stage/'best.pt'
        if best.exists():
            stages[stage]['best_checkpoint_sha256']=sha(best)
    status=state['state']
    if status=='RUNNING' and process.get('torchrun_proc_exists') is False:
        status='PROCESS_MISSING_INSPECT_LOGS'
    return dict(observed_management_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                reported_state=status,controller=state,process=process,latest_training_event=events[-1] if events else None,
                stages=stages,training_log=str(run/'logs/training.log'),test_evaluation=False,
                clock_note='Compute-node wall clock is approximately649s behind management at launch; preserve both clocks. Elapsed timings are node-local.',
                monitoring='Finite observation/publication only, every15min for at most72h; no training retries or new jobs.')


def publish(a):
    repo=Path(a.repo).resolve();run=Path(a.run).resolve();dest=repo/a.proposal
    snap=observe(run)
    (dest/'live_status.json').write_text(json.dumps(snap,indent=2)+'\n')
    event=snap['latest_training_event'] or {}
    lines=['# Actual execution status','',f"Observed management UTC: **{snap['observed_management_utc']}**.",'',
           f"State: **{snap['reported_state']}**. Source commit `{snap['controller']['commit']}`.",'',
           f"Host `{snap['controller']['host']}`, physical GPUs0/2 in shared pool-v2 mode; no PBS allocation or exclusive CPU/GPU ownership claimed.",
           f"Controller PID `{snap['controller']['controller_pid']}`, training torchrun PID `{snap['controller'].get('torchrun_pid')}`.",'',
           f"Latest recorded event: `{event.get('event')}`, stage `{event.get('stage')}`, epoch `{event.get('epoch')}`, batch `{event.get('batch')}/{event.get('batches')}`, loss `{event.get('loss')}`.",'',
           'Eight CPU tests PASS; real-TRAIN full-native-batch two-GPU CUDA smoke PASS. 128classification images/update; 64main+64auxiliary GDA images/update; 224px; seed17;50classification→TRAIN-SFD→50corrected-GDAepochs.',
           'Corrected differentiable class-conditioned GDA with frozen encoder BN is explicitly distinguished from the detached archived GDA and paper results. No new validation score exists until an epoch finishes. No TEST inference is attached.','',
           f"Continuous private log: `{snap['training_log']}`.",'',
           'Initial launch failed before GPU claim because compute nodes lack Git; a private byte-identical copy of management /usr/bin/git fixed source verification. Its hash and command are in execution_receipt.json. Model code/settings were unchanged.','',
           snap['clock_note'],'',snap['monitoring'],'',
           'See live_status.json for verified process observation, aggregate validation/group metrics, checkpoint hashes and completion/failure state. Review historical PROJECT_BLUEPRINT.md as a dated snapshot; this page is the execution update.','']
    (dest/'STATUS.md').write_text('\n'.join(lines))
    allowed=[a.proposal+'/STATUS.md',a.proposal+'/live_status.json']
    subprocess.run(['git','add','--',*allowed],cwd=repo,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo,text=True).splitlines()
    if not set(staged)<=set(allowed):
        raise RuntimeError('Unexpected staged files; do not publish someone else’s edits')
    subprocess.run(['git','diff','--cached','--check'],cwd=repo,check=True)
    if staged:
        subprocess.run(['git','commit','-m','Record actual Ding FF2 public training progress'],cwd=repo,check=True)
        subprocess.run(['git','push','origin',a.branch],cwd=repo,check=True,timeout=90)
    print(json.dumps(dict(observed=snap['observed_management_utc'],state=snap['reported_state'],event=event)),flush=True)
    return snap['reported_state']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--proposal',required=True)
    p.add_argument('--branch',required=True)
    p.add_argument('--once',action='store_true')
    p.add_argument('--hours',type=float,default=72)
    a=p.parse_args()
    lock=(Path(a.run)/'publisher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+a.hours*3600
    while True:
        status=publish(a)
        if a.once or status in ('COMPLETED_VERIFIED','FAILED','PROCESS_MISSING_INSPECT_LOGS') or time.monotonic()>deadline:
            break
        time.sleep(900)


if __name__=='__main__':
    main()
