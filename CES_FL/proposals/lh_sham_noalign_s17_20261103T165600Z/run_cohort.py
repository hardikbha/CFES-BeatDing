"""Exactly two independent arms, distinct UUIDs, one legitimate ngpus=2 allocation."""
import fcntl,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parents[2]
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/lh_sham_noalign_s17_20261103T165600Z')
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
resource=load('resource',SOURCE/'CES_FL/cluster/run_hierarchical_real.py')
m=load('model',SOURCE/'CES_FL/src/ces_fl_head_ablation.py')
lock=(ROOT/'cohort.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
job=os.environ['PBS_JOBID'];q=subprocess.check_output(['qstat','-f',job],text=True)
assert 'Resource_List.ngpus = 2' in q and 'gpu01:' in q
for arm in ('SHAM_FIX','NO_ALIGN'):
 c=m.load_config(HERE/(arm+'.json'));m.checked_inputs(c,ROOT/arm/'preflight');m.check_qualification(c,ROOT/arm/'review/qualification.json')
 assert not (ROOT/arm/'training').exists()
with resource.mutex():
 cards=resource.candidates();assert len(cards)>=2,'Two compatible distinct UUIDs unavailable; no launch/preemption'
 selected=cards[:2];assert selected[0]['uuid']!=selected[1]['uuid']
resource.write(ROOT/'COHORT_ALLOCATION.json',dict(job_id=job,ngpus=2,ncpus=10,mem_gb=64,host=os.uname().nodename,selected=selected,commit=resource.detached_head(SOURCE),route='PBS_ngpus2_plus_prior_explicit_shared_pool_UUID_leases',exclusive_physical_assignment_claimed=False))
children=[];start=time.monotonic()
for arm,card in zip(('SHAM_FIX','NO_ALIGN'),selected):
 env=os.environ.copy();env.update(CES_ABLATION_COHORT='SHAM_FIX_NO_ALIGN',CUDA_VISIBLE_DEVICES=card['uuid'])
 log=(ROOT/arm/'allocation.log').open('x')
 child=subprocess.Popen([sys.executable,'-u',str(HERE/'run_allocated.py'),arm],env=env,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT)
 children.append((arm,child,log));print('ARM_STARTED',arm,child.pid,card['uuid'],flush=True)
results={}
for arm,child,log in children:results[arm]=child.wait();log.close()
resource.write(ROOT/'COHORT_END.json',dict(job_id=job,exit_codes=results,elapsed_seconds=time.monotonic()-start,ngpus=2,no_test_access=True))
sys.exit(0 if all(v==0 for v in results.values()) else 1)
