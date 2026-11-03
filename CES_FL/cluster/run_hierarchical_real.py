"""Bounded CES-FL-only executor, explicit authorized shared pool-v2 leases.
No Ding model imports/stages, no FairForge-2, no TEST. Counts shared GPU walltime.
"""
import argparse,contextlib,fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,threading,time
from pathlib import Path

SOURCE=Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-hier-real-execution-20261103')
SCRIPT=SOURCE/'CES_FL/src/ces_fl_forensic_response.py'
COMMIT='3f5f3c16839bf4b96209bea3f968255b5f351ebb'
SHA='2549cb1345d1e2ee617e8a77ca288a0acbbfd05cf480f7b91a46b8627fcb74b2'
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/hierarchical_real_risk_s17_20261103_r1')
OLD=Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103')
PY='/home/sachin.chaudhary/.conda/envs/aiface/bin/python'
LOCKS=Path('/tmp/scf_gpu_locks');NAMES=('H_joint_real',)

CONFIG_SHA='63baed480e251945dba5ebb360e9ac1c7ea9fbc1cebf02acfef6abd6299fb3fd'
TASK_DEADLINE=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);t=p.with_name(p.name+f'.tmp.{os.getpid()}');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
budget=load(OLD/'allocated_stage.py','shared_budget_only')

def alive(pid,group=False):
 if not isinstance(pid,int) or pid<=0:return False
 try:(os.killpg if group else os.kill)(pid,0);return True
 except ProcessLookupError:return False
 except PermissionError:return True

@contextlib.contextmanager
def mutex():
 LOCKS.mkdir(exist_ok=True);lock=LOCKS/'.pool_mutex';until=time.monotonic()+30
 while True:
  try:lock.mkdir();break
  except FileExistsError:
   if time.monotonic()>until:raise RuntimeError('Pool mutex busy; never remove another lock')
   time.sleep(.2)
 try:yield
 finally:lock.rmdir()

def candidates(need=16000):
 import csv
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 cards=[dict(index=int(r[0]),uuid=r[1].strip(),free_mib=int(r[2]),util=int(r[3])) for r in csv.reader(text.splitlines())]
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True);apps=[]
 for r in csv.reader(text.splitlines()):
  try:apps.append((r[0].strip(),os.getpgid(int(r[1])),int(r[2])))
  except (ValueError,IndexError,ProcessLookupError,PermissionError):pass
 with budget.ledger() as ledger:
  occupied={s.get('card',{}).get('uuid') for s in ledger['gpu_sessions'] if s['state']=='RUNNING' and s.get('host')==os.uname().nodename and str(s['id']).startswith('hier_')}
 eligible=[]
 for c in cards:
  if c['uuid'] in occupied:continue
  directory=LOCKS/f"gpu{c['index']}";entries=[]
  if directory.exists():
   try:
    if not read(directory/'pool_owner.json').get('shared'):continue
    entries=[read(p) for p in directory.glob('task_*.json')];entries=[e for e in entries if alive(e.get('pid')) or alive(e.get('pgid'),True)]
   except (OSError,ValueError,TypeError):continue
  # Preserve existing Ding device capacity without invoking any Ding computation.
  if any(str(e.get('name','')).startswith('ding_ff2_public') or str(e.get('wid','')).startswith('ding_ff2_public') for e in entries):continue
  reserve=sum(max(0,int(e.get('need',0))-sum(m for uuid,pgid,m in apps if uuid==c['uuid'] and pgid==e.get('pgid'))) for e in entries)
  c['available_mib']=c['free_mib']-reserve
  if c['available_mib']>=need+4096:eligible.append(c)
 return sorted(eligible,key=lambda x:(x['util']>10,-x['available_mib']))

def reserve(sid,hours):
 if sid not in {'hier_smoke','hier_checks','hier_H_joint_real'}:raise RuntimeError('Unapproved stage/configuration')
 with budget.ledger() as d:
  if any(s['id']==sid for s in d['gpu_sessions']):raise RuntimeError('Duplicate reservation '+sid)
  committed=sum(s['reserved_gpu_hours'] if s['state'] in ['RUNNING','QUEUED'] else s.get('elapsed_seconds',0)*s['ngpus']/3600 for s in d['gpu_sessions'])
  if committed+hours>d['initial_total_gpu_hours']:raise RuntimeError('Would exceed shared24TOTAL GPU-hour budget')
  local=sum(s['reserved_gpu_hours'] if s['state'] in ['RUNNING','QUEUED'] else s.get('elapsed_seconds',0)*s['ngpus']/3600 for s in d['gpu_sessions'] if s['id'].startswith('hier_'))
  if local+hours>4.0:raise RuntimeError('Would exceed new-study4GPU-hour cap')
  d['gpu_sessions'].append(dict(id=sid,ngpus=1,reserved_gpu_hours=hours,elapsed_seconds=0.,state='QUEUED',scientific_source_sha256=SHA,scientific_commit=COMMIT,route='explicit_user_authorized_shared_pool_v2',pbs_allocation_claimed=False))

def claim(sid):
 until=time.monotonic()+3600
 while time.monotonic()<until:
  with mutex():
   options=candidates()
   if options:
    card=options[0];directory=LOCKS/f"gpu{card['index']}"
    if not directory.exists():directory.mkdir();write(directory/'pool_owner.json',dict(shared=True,v=2,pid=0,wid='pool_v2_shared'))
    entry=directory/f'task_{sid}.json'
    if entry.exists():raise RuntimeError('Own-name lease already exists')
    write(entry,dict(name=sid,wid=sid,pid=os.getpid(),pgid=None,cls='reader',need=16000,time=time.time()));return card,entry
  time.sleep(20)
 raise RuntimeError('No compatible explicitly leased device within1hour; no preemption')

def command(args,log,card,entry,script=SCRIPT):
 bootstrap="import sys,torch,runpy,json,os;assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.is_bf16_supported();assert 'H100' in torch.cuda.get_device_name(0);torch.cuda.set_per_process_memory_fraction(.16);print(json.dumps(dict(event='CUDA_BINDING',uuid=os.environ['CUDA_VISIBLE_DEVICES'],device=torch.cuda.get_device_name(0),bf16=True,torch=str(torch.__version__))),flush=True);sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=card['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
 cmd=[PY,'-u','-c',bootstrap,str(script),*args]
 with Path(log).open('x') as f:
  f.write(json.dumps(dict(command=cmd,scientific_commit=COMMIT,source_sha256=SHA,gpu=card,lease=str(entry),pbs_allocation_claimed=False))+'\n');f.flush()
  child=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  with mutex():e=read(entry);e['pgid']=child.pid;write(entry,e)
  with budget.ledger() as ledger:
   session=next(s for s in ledger['gpu_sessions'] if s['id']==entry.stem[5:]);session.update(child_pid=child.pid,log=str(log))
  try:
   code=child.wait(timeout=max(1.,TASK_DEADLINE-time.monotonic()))
  except subprocess.TimeoutExpired:
   # Only this newly launched child process group; never an existing job.
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   raise RuntimeError('Own task resource cap reached; preserve checkpoint/log; no retry')
  if code!=0:raise RuntimeError('Command failed; preserve '+str(log))

def detached_head(source):
 pointer=Path(source)/'.git'
 if pointer.is_file():
  text=pointer.read_text().strip()
  if not text.startswith('gitdir: '):raise RuntimeError('Invalid worktree pointer')
  gitdir=Path(text[8:]);gitdir=gitdir if gitdir.is_absolute() else Path(source)/gitdir
 else:gitdir=pointer
 head=(gitdir/'HEAD').read_text().strip()
 if len(head)!=40 or any(ch not in '0123456789abcdef' for ch in head):raise RuntimeError('Execution HEAD is not detached')
 return head

def verify_source():
 if sha(SCRIPT)!=SHA or detached_head(SOURCE)!=COMMIT:raise RuntimeError('Pinned scientific source mismatch')
 # The nodes lack git. The launch host verified the whole checkout clean;
 # this node independently checks detached HEAD and the standalone source bytes.
 receipt=read(ROOT/'SOURCE_CHECKOUT.json')
 if receipt.get('commit')!=COMMIT or receipt.get('source_sha256')!=SHA or receipt.get('git_status_porcelain')!='':raise RuntimeError('Missing clean-checkout provenance')

def ready(name,qualification=True):
 if name not in NAMES:raise RuntimeError('Only the one preregistered CES-FL candidate is allowed')
 if sha(ROOT/name/'config.json')!=CONFIG_SHA:raise RuntimeError('Fixed candidate config bytes changed')
 r=ROOT/name;c=read(r/'config.json');digest=hashlib.sha256(json.dumps(c,sort_keys=True,allow_nan=False).encode()).hexdigest();pre=read(r/'preflight/PREFLIGHT.json')
 if c['dataset_id']!='FFPP_DING_PUBLIC' or c['seed']!=17 or c['epochs']!=10 or c['error_control']!='hierarchical_real' or c['alignment_warmup_epochs']!=0:raise RuntimeError('Candidate scientific configuration changed')
 if pre['status']!='PASS' or pre['source_sha256']!=SHA or pre['config_digest']!=digest:raise RuntimeError('Fresh bound CPU input preflight required')
 if qualification:
  m=load(SCRIPT,'qualification_check');m.check_qualification(c,r/'qualification.json')
 return c,digest

def verified(name):
 r=ROOT/name/'training'
 if not (r/'DONE').exists():return False
 result=read(r/'RUN_VERIFIED.json');_,digest=ready(name)
 if result['status']!='COMPLETED_VERIFIED' or result['source_sha256']!=SHA or result['config_digest']!=digest or result['epochs_completed']!=10 or result['selected_checkpoint_sha256']!=sha(r/'best.pt') or result['test_evaluated']:raise RuntimeError('Completion provenance/hash mismatch')
 return True

def main():
 global TASK_DEADLINE
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['smoke','checks','train'],required=True);p.add_argument('--candidate',choices=NAMES);p.add_argument('--diagnostic-script');a=p.parse_args();verify_source()
 if a.task=='train' and not a.candidate:raise RuntimeError('Explicit candidate required')
 sid='hier_'+(a.candidate if a.task=='train' else a.task)
 lock=(ROOT/(sid+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if a.task!='smoke':
  for name in ([a.candidate] if a.task=='train' else NAMES):ready(name)
 if a.task=='train':
  if verified(a.candidate):print('SKIP_VERIFIED_COMPLETED');return
  if (ROOT/a.candidate/'training').exists():raise RuntimeError('Existing interrupted run requires an explicit compatible resume operator; do not overwrite')
  smoke=read(ROOT/'software_cuda/SMOKE_RESULT.json');dry=read(ROOT/a.candidate/'dry/DRY_RUN.json');_,digest=ready(a.candidate)
  if smoke['status']!='PASS' or smoke['source_sha256']!=SHA or not smoke['cuda_tested'] or dry['status']!='PASS' or dry['source_sha256']!=SHA or dry['config_digest']!=digest or dry['device']!='cuda':raise RuntimeError('Current source/config CUDA checks required')
  real=read(ROOT/'real_train_check/REAL_TRAIN_CHECK.json')
  if real['status']!='PASS' or real['source_sha256']!=SHA or real['config_digest']!=digest or real['device']!='cuda' or real['real_component_gradient_norm']<=0 or real['test_read']:raise RuntimeError('Changed real component requires fresh TRAIN-threshold CUDA gradient check')
 reserve(sid,3.5 if a.task=='train' else .25);started=None;entry=None;status='FAILED';stop=threading.Event();t=None
 try:
  card,entry=claim(sid);started=time.monotonic();TASK_DEADLINE=started+(3.5 if a.task=='train' else .25)*3600-30
  with budget.ledger() as d:
   row=next(s for s in d['gpu_sessions'] if s['id']==sid);row.update(state='RUNNING',host=os.uname().nodename,controller_pid=os.getpid(),card=card)
  def heartbeat():
   while not stop.wait(15):
    with budget.ledger() as d:next(s for s in d['gpu_sessions'] if s['id']==sid)['elapsed_seconds']=time.monotonic()-started
  t=threading.Thread(target=heartbeat,daemon=True);t.start()
  if a.task=='smoke':command(['smoke','--device','cuda','--out',str(ROOT/'software_cuda')],ROOT/'software_cuda.log',card,entry)
  elif a.task=='checks':
   for name in NAMES:
    r=ROOT/name;command(['dry-run','--config',str(r/'config.json'),'--preflight',str(r/'preflight'),'--arm','D','--device','cuda','--out',str(r/'dry')],r/'dry.log',card,entry)
   if not a.diagnostic_script:raise RuntimeError('Changed real-component TRAIN gradient check is required')
   command(['--source',str(SOURCE),'--run',str(ROOT),'--out',str(ROOT/'real_train_check')],ROOT/'diagnostics.log',card,entry,Path(a.diagnostic_script))
  else:
   r=ROOT/a.candidate;out=r/'training';base=['train','--config',str(r/'config.json'),'--preflight',str(r/'preflight'),'--arm','D','--device','cuda','--software-receipt',str(ROOT/'software_cuda/SMOKE_RESULT.json'),'--dry-run-receipt',str(r/'dry/DRY_RUN.json'),'--qualification',str(r/'qualification.json'),'--out',str(out)]
   command(base,r/'train_initial.log',card,entry)
   for hours in [3]:
    if verified(a.candidate):break
    info=read(out/'STATUS.json')
    if info['status']!='INCOMPLETE_EPOCH_BOUNDARY_STOP' or not (out/'last.pt').exists():raise RuntimeError('Not a documented compatible epoch-boundary interruption')
    write(r/f'resume{hours}_PLAN.json',dict(source_commit=COMMIT,source_sha256=SHA,config_digest=ready(a.candidate)[1],same_output=True,same_qualification=True,completed_epochs=info['completed_epochs'],planned_epochs=10,new_cumulative_allowance_hours=hours,no_test_tuning=True))
    command(base+['--resume','--budget-hours',str(hours)],r/f'train_resume{hours}.log',card,entry)
   if not verified(a.candidate):raise RuntimeError('Still incomplete after documented3hour cumulative allowance; preserve and report')
  status='COMPLETED'
 finally:
  stop.set()
  if t is not None:t.join(timeout=20)
  elapsed=time.monotonic()-started if started is not None else 0.
  with budget.ledger() as d:next(s for s in d['gpu_sessions'] if s['id']==sid).update(state=status,elapsed_seconds=elapsed)
  if entry is not None:
   with mutex():
    if entry.exists() and read(entry).get('pid')==os.getpid() and not alive(read(entry).get('pgid'),True):entry.unlink()
  write(ROOT/(sid+'_END.json'),dict(status=status,elapsed_seconds=elapsed,gpu_hours=elapsed/3600,source_commit=COMMIT,source_sha256=SHA,pbs_allocation_claimed=False,test_read=False))
if __name__=='__main__':main()
