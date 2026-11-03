"""Finite publisher for the two already submitted jobs. No execution/release/retry."""
import datetime,hashlib,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/lh_sham_noalign_s17_20261103T165600Z')
REL='CES_FL/results/lh_sham_noalign_s17_20261103T165600Z'
previous=None;end=time.monotonic()+3*3600
while time.monotonic()<end:
 states=[];finished=[]
 for arm in ('SHAM_FIX','NO_ALIGN'):
  r=ROOT/arm;job=json.loads((r/'SUBMISSION.json').read_text())['stdout'].strip()
  q=subprocess.run(['qstat','-x','-f','-F','json',job],capture_output=True,text=True)
  state=json.loads(q.stdout)['Jobs'][job]['job_state'] if not q.returncode else 'UNVERIFIED'
  files=['preflight/PREFLIGHT.json','review/qualification.json','smoke_cuda/SMOKE_RESULT.json','dry/DRY_RUN.json','training/history.json','training/RUN_VERIFIED.json','ALLOCATION_END.json','SCHEDULER_ACCOUNTING.json','comparison/COMPARISON.json']
  states.append((arm,state,[(f,hashlib.sha256((r/f).read_bytes()).hexdigest()) for f in files if (r/f).exists()]))
  finished.append(state=='F')
 signature=repr(states)
 if signature!=previous:
  subprocess.run([sys.executable,str(HERE/'publish_snapshot.py')],cwd=REPO,check=True)
  subprocess.run(['git','add','--',REL],cwd=REPO,check=True)
  dirty=subprocess.run(['git','diff','--cached','--quiet','--',REL],cwd=REPO)
  if dirty.returncode==1:
   subprocess.run(['git','commit','-m','Record independent L_H ablation checks and epoch progress','--',REL],cwd=REPO,check=True)
   subprocess.run(['git','push','origin','experiment/lh-sham-noalign-s17-20261103'],cwd=REPO,check=True)
  previous=signature
  print(datetime.datetime.now(datetime.timezone.utc).isoformat(),[(s[0],s[1]) for s in states],flush=True)
 if all(finished):break
 time.sleep(30)
else:print('Finite publication window ended; no execution change',flush=True)
