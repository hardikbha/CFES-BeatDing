"""Exactly one three-epoch run, actual PBSngpus1+UUID lease and shared ledger."""
import fcntl,importlib.util,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parents[2]
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/lh_low_lr3_s17_20261103T140700Z')
SHA='08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177'
CFG_SHA='7ac0d405161210e11846ce24af420a77ebdb16074cf39df8d0f80e5742fe6085'
SID='lh_low_lr3_s17_20261103';CAP=1.5
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
resource=load('resource_only',SOURCE/'CES_FL/cluster/run_hierarchical_real.py');budget=resource.budget
write=resource.write;read=resource.read
def main():
 job=os.environ.get('PBS_JOBID');assert job
 lock=(ROOT/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 started=time.monotonic();stop=threading.Event();child=None;entry=None;status='FAILED'
 with budget.ledger() as d:
  row=next(s for s in d['gpu_sessions'] if s['id']==SID);assert row['state']=='QUEUED' and row['reserved_gpu_hours']==CAP
  row.update(state='RUNNING',pbs_job_id=job,pid=os.getpid(),host=os.uname().nodename)
 def heartbeat():
  while not stop.wait(15):
   with budget.ledger() as d:next(s for s in d['gpu_sessions'] if s['id']==SID)['elapsed_seconds']=time.monotonic()-started
 thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
 try:
  qt=subprocess.check_output(['/opt/pbs/bin/qstat','-f',job],text=True,timeout=30);(ROOT/'allocation_qstat_PRIVATE.txt').write_text(qt)
  assert 'Resource_List.ngpus = 1' in qt and 'gpu02:' in qt
  binding=read(ROOT/'SOURCE_CHECKOUT.json');commit=resource.detached_head(SOURCE)
  assert binding['commit']==commit and binding['git_status_porcelain']=='' and resource.sha(SOURCE/'CES_FL/src/ces_fl_head_ablation.py')==SHA
  assert resource.sha(HERE/'config.json')==CFG_SHA
  for name,expected in binding['proposal_file_sha256'].items():assert resource.sha(HERE/name)==expected
  m=load('plain',SOURCE/'CES_FL/src/ces_fl_head_ablation.py');c=m.load_config(HERE/'config.json')
  m.checked_inputs(c,ROOT/'preflight');m.check_qualification(c,ROOT/'review/qualification.json')
  assert read(ROOT/'REQUEST_CHECK.json')['status']=='PASS' and read(ROOT/'train_drift/TRAIN_DRIFT.json')['status']=='PASS'
  assert not (ROOT/'training').exists()
  with resource.mutex():
   cards=resource.candidates();supplied=os.environ.get('CUDA_VISIBLE_DEVICES','').strip()
   if supplied:cards=[x for x in cards if supplied in (str(x['index']),x['uuid'])]
   if not cards:raise RuntimeError('No eligible user-authorized shared UUID; stop without preemption')
   card=cards[0];folder=resource.LOCKS/f"gpu{card['index']}"
   if not folder.exists():folder.mkdir();write(folder/'pool_owner.json',dict(shared=True,v=2,pid=0,wid='pool_v2_shared'))
   entry=folder/f'task_{SID}.json';assert not entry.exists()
   write(entry,dict(name=SID,wid=SID,pid=os.getpid(),pgid=None,cls='reader',need=16000,time=time.time(),pbs_job_id=job))
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=card['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
  init="import torch,os,json;assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported();assert 'H100' in torch.cuda.get_device_name(0);print(json.dumps(dict(uuid=os.environ['CUDA_VISIBLE_DEVICES'],device=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=torch.version.cuda,bf16=True)),flush=True)"
  with (ROOT/'CUDA_BINDING.log').open('x') as f:subprocess.run([sys.executable,'-c',init],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  write(ROOT/'ALLOCATION.json',dict(pbs_job_id=job,host=os.uname().nodename,ngpus=1,ncpus=6,mem_gb=32,card=card,cuda_visible_devices=card['uuid'],scheduler_supplied_binding=supplied or None,physical_binding_route='explicit_prior_user_authorized_pool_v2_UUID_lease',exclusive_physical_assignment_claimed=False,source_commit=commit,source_sha256=SHA,config_sha256=CFG_SHA,controller_pid=os.getpid()))
  with budget.ledger() as d:next(s for s in d['gpu_sessions'] if s['id']==SID).update(card=card,log=str(ROOT/'allocation.log'))
  child=subprocess.Popen(['bash',str(HERE/'commands.sh'),'gpu'],cwd=SOURCE,env=env,start_new_session=True)
  with resource.mutex():e=read(entry);e['pgid']=child.pid;write(entry,e)
  with budget.ledger() as d:next(s for s in d['gpu_sessions'] if s['id']==SID)['child_pid']=child.pid
  try:code=child.wait(timeout=max(1,CAP*3600-(time.monotonic()-started)-30))
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   raise RuntimeError('Own allocation cap reached; preserve incomplete run, no automatic extension')
  if code:raise RuntimeError('Explicit check/train sequence failed; no retry or extension')
  out=ROOT/'training';r=read(out/'RUN_VERIFIED.json');assert (out/'DONE').is_file() and r['status']=='COMPLETED_VERIFIED' and r['epochs_completed']==3 and r['source_sha256']==SHA and r['config_digest']==m.digest(c) and r['selected_checkpoint_sha256']==m.sha(out/'best.pt') and r['test_evaluated'] is False
  status='COMPLETED'
 finally:
  stop.set();thread.join(timeout=20);elapsed=time.monotonic()-started
  with budget.ledger() as d:next(s for s in d['gpu_sessions'] if s['id']==SID).update(state=status,elapsed_seconds=elapsed)
  if entry is not None and (child is None or child.poll() is not None):
   with resource.mutex():
    if entry.exists() and read(entry).get('pid')==os.getpid():entry.unlink()
  write(ROOT/'ALLOCATION_END.json',dict(status=status,elapsed_seconds=elapsed,gpu_hours=elapsed/3600,pbs_job_id=job,test_read=False))
if __name__=='__main__':main()
