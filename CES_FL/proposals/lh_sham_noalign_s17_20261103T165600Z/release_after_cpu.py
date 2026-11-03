"""Release only the two authorized, held jobs after their fresh CPU gates pass.

No training submission, retries, input edits, approval creation or source mutation.
"""
import datetime,fcntl,importlib.util,json,os,subprocess,time
from pathlib import Path
SOURCE=Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-lh-sham-noalign-execution-20261103')
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/lh_sham_noalign_s17_20261103T165600Z')
PROPOSAL=SOURCE/'CES_FL/proposals/lh_sham_noalign_s17_20261103T165600Z'
spec=importlib.util.spec_from_file_location('pinned',SOURCE/'CES_FL/src/ces_fl_head_ablation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
lock=(ROOT/'release.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
pending={'SHAM_FIX':'45350.mgmt01','NO_ALIGN':'45351.mgmt01'}
deadline=time.monotonic()+3600
while pending and time.monotonic()<deadline:
 for arm,job in list(pending.items()):
  r=ROOT/arm
  if not (r/'review/qualification.json').exists():continue
  c=m.load_config(PROPOSAL/(arm+'.json'));m.checked_inputs(c,r/'preflight');m.check_qualification(c,r/'review/qualification.json')
  req=json.loads((r/'REQUEST_CHECK.json').read_text());assert req['status']=='PASS' and req['source_sha256']==m.sha(m.__file__) and req['config_digest']==m.digest(c)
  assert m.sha(m.__file__)=='1b6dc1b0e6d2b2220da897500a4d1bffb7764280df642abf6befdd1cf511ed01'
  state=json.loads(subprocess.check_output(['qstat','-f','-F','json',job],text=True))['Jobs'][job]
  assert state['job_state']=='H' and state['Resource_List']['ngpus']==1
  record=dict(job_id=job,arm=arm,source_sha256=m.sha(m.__file__),config_digest=m.digest(c),qualification_sha256=m.sha(r/'review/qualification.json'),observed_management_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  with (r/'RELEASE.json').open('x') as f:
   result=subprocess.run(['qrls',job],capture_output=True,text=True)
   record.update(command=result.args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr);json.dump(record,f,indent=2)
  if result.returncode:raise RuntimeError('Release failed; no retry')
  print('RELEASED_AFTER_VERIFIED_CPU_GATES',arm,job,flush=True);del pending[arm]
 if pending:time.sleep(15)
if pending:raise RuntimeError('CPU gates not ready within one hour; held jobs not released: '+str(pending))
