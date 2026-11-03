"""Authorized finite CPU-analysis execution controller; no training or TEST."""
import datetime,fcntl,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1')
SOURCE=Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103')
SHA='0eeafd838da3ca158336dcfe3b8195842dd15bc9fecd67ebbaf2ea03ba3c2680'
def read(p):return json.loads(Path(p).read_text())
def write(x):
 x.update(observed_management_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_launch_capability=False,test_read=False);p=ROOT/'POSTHOC_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p);print(json.dumps(x),flush=True)
def main():
 lock=(ROOT/'posthoc_execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);deadline=time.monotonic()+24*3600
 while True:
  complete=[]
  for name in ['E_support','F_support_ramp3']:
   r=ROOT/name/'training';p=r/'RUN_VERIFIED.json'
   if (r/'DONE').exists() and p.exists():
    d=read(p);assert d['status']=='COMPLETED_VERIFIED' and d['source_sha256']==SHA and d['epochs_completed']==10 and not d['test_evaluated'];assert hashlib.sha256((r/'best.pt').read_bytes()).hexdigest()==d['selected_checkpoint_sha256'];complete.append(name)
  if len(complete)==2:break
  ledger=read('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/budget_ledger.json')
  failures=[s['id'] for s in ledger['gpu_sessions'] if s['id'] in ['support_E_support','support_F_support_ramp3'] and s['state']=='FAILED']
  if failures:write(dict(state='BLOCKED_INCOMPLETE_OR_FAILED_CANDIDATE',failures=failures,completed=complete));return
  write(dict(state='WAITING_NATIVE_TEN_EPOCH_COMPLETIONS',completed=complete))
  if time.monotonic()>deadline:write(dict(state='FINITE_DEADLINE_NO_ANALYSIS_LAUNCHED'));return
  time.sleep(30)
 record=ROOT/'analysis_submission.json'
 if record.exists():raise RuntimeError('Analysis submission already recorded; never duplicate')
 script=ROOT/'validation_analysis.pbs'
 body='''#!/bin/bash
#PBS -S /bin/bash
#PBS -N cesfl_VAL_CI_s17
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/validation_analysis_pbs.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
cd /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1
exec /home/sachin.chaudhary/.conda/envs/aiface/bin/python -u runtime/support_validation_decision.py --source /home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103 --run /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1 --out /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/validation_analysis
'''
 with script.open('x') as f:f.write(body)
 p=subprocess.run(['qsub',str(script)],capture_output=True,text=True);job=p.stdout.strip() if p.returncode==0 else None
 with record.open('x') as f:f.write(json.dumps(dict(command=['qsub',str(script)],returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,pbs_job_id=job,ncpus=4,requested_ngpus=0,task='CPU paired fixed-VAL uncertainty only; no model training or TEST'),indent=2)+'\n')
 if p.returncode:write(dict(state='CPU_ANALYSIS_SUBMISSION_FAILED',error=p.stderr));return
 while time.monotonic()<deadline:
  decision=ROOT/'validation_analysis/DECISIONS.json'
  if decision.exists():
   d=read(decision);assert d['status']=='VALIDATION_ANALYSIS_COMPLETED' and d['source_sha256']==SHA and not d['test_read'];write(dict(state='COMPLETED_ANALYSIS_NO_AUTOMATIC_PROMOTION',pbs_job_id=job,decisions={n:x['decision'] for n,x in d['results'].items()}));return
  p=subprocess.run(['qstat','-xf',job],capture_output=True,text=True,timeout=25);m=re.search(r'job_state\s*=\s*([A-Z])',p.stdout)
  if m and m.group(1)=='F':write(dict(state='CPU_ANALYSIS_FAILED_OR_MISSING_DECISIONS',pbs_job_id=job));return
  write(dict(state='CPU_ANALYSIS_'+(m.group(1) if m else 'UNVERIFIED'),pbs_job_id=job));time.sleep(30)
 write(dict(state='FINITE_DEADLINE_ANALYSIS_STATUS_UNVERIFIED',pbs_job_id=job))
if __name__=='__main__':main()
