"""Finite CPU-analysis execution controller; never launches training or TEST."""
import datetime,fcntl,hashlib,json,os,re,subprocess,time
from pathlib import Path
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/hierarchical_real_risk_s17_20261103_r1')
SHA='2549cb1345d1e2ee617e8a77ca288a0acbbfd05cf480f7b91a46b8627fcb74b2'
def read(p):return json.loads(Path(p).read_text())
def status(**x):
    x.update(observed_management_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),can_train=False,test_read=False)
    p=ROOT/'POSTHOC_STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p);print(json.dumps(x),flush=True)
def main():
    lock=(ROOT/'posthoc.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        ledger=read('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/budget_ledger.json')
        sessions=[s for s in ledger['gpu_sessions'] if s['id']=='hier_H_joint_real']
        if sessions and sessions[-1]['state']=='FAILED':status(state='BLOCKED_FAILED_OR_INCOMPLETE_CANDIDATE');return
        r=ROOT/'H_joint_real/training';verified=r/'RUN_VERIFIED.json'
        if sessions and sessions[-1]['state']=='COMPLETED' and (r/'DONE').exists() and verified.exists():
            d=read(verified);assert d['status']=='COMPLETED_VERIFIED' and d['epochs_completed']==10 and d['source_sha256']==SHA and not d['test_evaluated']
            assert hashlib.sha256((r/'best.pt').read_bytes()).hexdigest()==d['selected_checkpoint_sha256'];break
        status(state='WAITING_NATIVE_COMPLETION_AND_GPU_RELEASE');time.sleep(30)
    else:status(state='FINITE_DEADLINE_NO_ANALYSIS_LAUNCHED');return
    record=ROOT/'analysis_submission.json'
    if record.exists():raise RuntimeError('Existing analysis submission; do not duplicate')
    jobscript=ROOT/'validation_analysis.pbs'
    body='''#!/bin/bash
#PBS -S /bin/bash
#PBS -N cesfl_H_VAL_s17
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /home/sachin.chaudhary/ces_fl_runs/hierarchical_real_risk_s17_20261103_r1/validation_analysis_pbs.log
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
cd /home/sachin.chaudhary/ces_fl_runs/hierarchical_real_risk_s17_20261103_r1
exec /home/sachin.chaudhary/.conda/envs/aiface/bin/python -u runtime/analyze_hier_real_validation.py --source /home/sachin.chaudhary/ces_fl_sources/cesfl-hier-real-execution-20261103 --run /home/sachin.chaudhary/ces_fl_runs/hierarchical_real_risk_s17_20261103_r1 --out /home/sachin.chaudhary/ces_fl_runs/hierarchical_real_risk_s17_20261103_r1/validation_analysis --plotter /home/sachin.chaudhary/ces_fl_runs/hierarchical_real_risk_s17_20261103_r1/runtime/plot_hier_real_metrics.py --exporter /home/sachin.chaudhary/ces_fl_runs/hierarchical_real_risk_s17_20261103_r1/runtime/export_cesfl_selected_failure_audit.py
'''
    with jobscript.open('x') as f:f.write(body)
    p=subprocess.run(['qsub',str(jobscript)],capture_output=True,text=True)
    with record.open('x') as f:f.write(json.dumps(dict(command=['qsub',str(jobscript)],returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,ngpus=0,ncpus=4,training_or_test=False),indent=2)+'\n')
    if p.returncode:status(state='CPU_SUBMISSION_FAILED',error=p.stderr);return
    job=p.stdout.strip()
    while time.monotonic()<deadline:
        dec=ROOT/'validation_analysis/DECISIONS.json'
        if dec.exists():
            d=read(dec);assert d['source_sha256']==SHA and not d['test_read'];status(state='COMPLETED_NO_RETRY_OR_PROMOTION',decision=d['decision'],pbs_job_id=job);return
        p=subprocess.run(['qstat','-xf',job],capture_output=True,text=True,timeout=25);m=re.search(r'job_state\s*=\s*(\w)',p.stdout)
        if m and m.group(1)=='F':status(state='FAILED_CPU_ANALYSIS_OR_MISSING_DECISION',pbs_job_id=job);return
        status(state='CPU_ANALYSIS_'+(m.group(1) if m else 'UNVERIFIED'),pbs_job_id=job);time.sleep(30)
    status(state='FINITE_DEADLINE_ANALYSIS_UNVERIFIED',pbs_job_id=job)
if __name__=='__main__':main()
