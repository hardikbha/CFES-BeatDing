"""Finite read-only CES-FL observer/publication; never launches jobs or reviews."""
import argparse,datetime,fcntl,hashlib,json,shlex,subprocess,time
from pathlib import Path
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1')
REPO=Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-s17-20261103')
REL='CES_FL/results/support_warmup_s17_20261103_r1'
BRANCH='experiment/cesfl-support-warmup-s17-20261103'
SHA='0eeafd838da3ca158336dcfe3b8195842dd15bc9fecd67ebbaf2ea03ba3c2680'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def observe():
 dest=REPO/REL;dest.mkdir(exist_ok=True);ledger=read('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/budget_ledger.json');live={}
 sessions=[s for s in ledger['gpu_sessions'] if s['id'].startswith('support_')]
 for host in ('gpu01','gpu02'):
  pids=[s['child_pid'] for s in sessions if s['state']=='RUNNING' and s.get('host')==host and s.get('child_pid')]
  if not pids:continue
  code='from pathlib import Path;import json;print(json.dumps({str(p):(Path("/proc/"+str(p)+"/cmdline").read_bytes().replace(bytes([0]),b" ").decode(errors="replace") if Path("/proc/"+str(p)+"/cmdline").exists() else "") for p in '+repr(pids)+'}))'
  try:
   result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=25);live[host]=json.loads(result.stdout) if result.returncode==0 else dict(state='SSH_FAILED_UNVERIFIED')
  except subprocess.TimeoutExpired:live[host]=dict(state='SSH_TIMEOUT_UNVERIFIED')
 candidates={}
 for name in ['E_support','F_support_ramp3']:
  root=ROOT/name;out=root/'training';c=read(root/'config.json');digest=hashlib.sha256(json.dumps(c,sort_keys=True,allow_nan=False).encode()).hexdigest();pre=root/'preflight/PREFLIGHT.json';q=root/'qualification.json';dry=root/'dry/DRY_RUN.json';hist=read(out/'history.json') if (out/'history.json').exists() else []
  selected=max(hist,key=lambda x:x['validation']['overall']['auc']) if hist else None;run_sessions=[s for s in sessions if s['id']=='support_'+name];running=[s for s in run_sessions if s['state']=='RUNNING'];state='BLOCKED_INPUTS';reason='source-bound input/qualification/CUDA gates pending'
  if pre.exists() and q.exists() and dry.exists():state='PENDING';reason=None
  if running:
   s=running[-1];cmd=live.get(s['host'],{}).get(str(s.get('child_pid')),'');state='RUNNING' if isinstance(cmd,str) and 'ces_fl_forensic_response.py' in cmd and str(root/'config.json') in cmd else 'RUNNING_UNVERIFIED';reason=None
  elif (out/'DONE').exists() and (out/'RUN_VERIFIED.json').exists():
   r=read(out/'RUN_VERIFIED.json');state='COMPLETED_VERIFIED' if r['status']=='COMPLETED_VERIFIED' and r['source_sha256']==SHA and r['config_digest']==digest and r['epochs_completed']==10 and r['selected_checkpoint_sha256']==sha(out/'best.pt') and not r['test_evaluated'] else 'COMPLETION_PROVENANCE_MISMATCH';reason=None
  elif any(s['state']=='FAILED' for s in run_sessions):state='FAILED';reason='See immutable attempt logs; no automatic failed-run retry'
  elif (out/'STATUS.json').exists():state=read(out/'STATUS.json')['status'];reason=None
  item=dict(state=state,blocker=reason,epochs_completed=len(hist),planned_epochs=10,config_file_sha256=sha(root/'config.json'),config_digest=digest,
    preflight=read(pre) if pre.exists() else None,qualification_sha256=sha(q) if q.exists() else None,
    review_scope=read(root/'operator_preview/REVIEW_SUMMARY.json') if (root/'operator_preview/REVIEW_SUMMARY.json').exists() else None,
    backward_check=read(dry) if dry.exists() else None,selected_epoch=selected['epoch'] if selected else None,
    selected_validation=selected['validation'] if selected else None,completed_receipt=read(out/'RUN_VERIFIED.json') if (out/'RUN_VERIFIED.json').exists() else None,
    checkpoint_sha256=sha(out/'best.pt') if (out/'best.pt').exists() else None,sessions=run_sessions,
    epoch_exposure=[{k:r.get(k) for k in ['epoch','unique_originals','original_real_pairs','originals_once_each','original_real_pairs_by_axis_group','auxiliary_original_draws_by_axis_group_label','auxiliary_real_pair_draws_by_axis_group','recorded_contributor_support_histogram','alignment_multiplier','epoch_seconds']} for r in hist])
  candidates[name]=item
 controls={}
 for a in 'ABCD':
  p=Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_overlapfix_20261103_r1_ffpp_s17')/a/'RUN_VERIFIED.json';controls[a]=read(p)
 result=dict(observed_management_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scientific_commit='8d5f0996a3f1cc02fbeaff9d25c1d95f714e8e77',source_sha256=SHA,candidates=candidates,existing_CESFL_controls=controls,
  gpu_ledger=ledger,actual_process_observations=live,software_cuda=read(ROOT/'software_cuda/SMOKE_RESULT.json') if (ROOT/'software_cuda/SMOKE_RESULT.json').exists() else None,
  decisions=read(ROOT/'validation_analysis/DECISIONS.json') if (ROOT/'validation_analysis/DECISIONS.json').exists() else None,
  execution_monitor=read(ROOT/'POSTHOC_STATUS.json') if (ROOT/'POSTHOC_STATUS.json').exists() else None,
  route='Explicit user-authorized pool-v2 shared UUID leases; no PBS or exclusive-GPU allocation claimed',
  no_new_ding_model_computation=True,new_FairForge2_work_deferred=True,test_read=False)
 (dest/'LIVE_STATE.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# CES-FL support-aware seed17 pilots','',f"Observed management UTC: {result['observed_management_utc']}",'',
 'Scientific commit8d5f0996a3f1cc02fbeaff9d25c1d95f714e8e77; source SHA2560eeafd838da3ca158336dcfe3b8195842dd15bc9fecd67ebbaf2ea03ba3c2680. Both candidates seed17,tenepochs. Existing CES-FL controls reused; no new Ding model computation, no new FF2 experiments, no TEST.','',
 result['route']+'. Existing Ding jobs untouched.','',f"Shared GPU budget used{ledger['consumed_gpu_hours']:.4f}/24TOTAL hours; unspent conservative reservations{ledger['reserved_gpu_hours']:.4f}h. Both GPUs counted by full task-held walltime even when shared.",'',
 '|Candidate|Actual state|Complete epochs|Selected epoch|VAL AUC (%)|Intersection F_FPR (fraction)|Gender F_FPR|Age F_FPR|','|---|---|---:|---:|---:|---:|---:|---:|']
 for n,d in candidates.items():
  v=d['selected_validation'];values=[f"{100*v['overall']['auc']:.6f}",f"{v['axes']['gender_race']['F_FPR']:.6f}",f"{v['axes']['gender']['F_FPR']:.6f}",f"{v['axes']['age_bin']['F_FPR']:.6f}"] if v else ['—']*4
  lines.append(f"|{n}|{d['state']}|{d['epochs_completed']}/10|{d['selected_epoch'] or '—'}|"+'|'.join(values)+'|')
 lines+=['','Values are selected-so-far VAL until native ten-epoch completion/replay. All supported axes, FPR/FNR, group support, achieved calibration, pair exposure and hashes are in LIVE_STATE.json. Gates are learned weights; maps are signed local contributions. AI visual review qualifies representative procedural recipes, not biological identity or real-forgery validity.','',
 'Prospective decision rules and comparison limits: ../../proposals/support_warmup_s17_20261103_r1/README.md and DING_COMPARISON_AUDIT.md. Female-Black validation has only30real images/two recorded components. No statistical superiority or paper TEST win follows from a favorable pilot metric.','',
 'E changes the error-control package; F adds a3epoch alignment ramp. A final schedule is separately recorded/compute-gated; no automatic promotion or TEST inference. Observer runs every10minutes for at most24hours and cannot train, submit jobs or generate reviews.']
 if result['decisions']:lines+=['','Preregistered decisions: '+json.dumps({n:x['decision'] for n,x in result['decisions']['results'].items()})]
 (dest/'STATUS.md').write_text('\n'.join(lines)+'\n')
 allowed=[REL+'/LIVE_STATE.json',REL+'/STATUS.md'];subprocess.run(['git','add','--',*allowed],cwd=REPO,check=True);staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).splitlines()
 if not set(staged)<=set(allowed):raise RuntimeError('Unexpected staged files; preserve other work')
 subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
 if staged:
  subprocess.run(['git','-c','gc.auto=0','commit','-m','Publish actual seed17 CES-FL candidate stages and shared GPU budget'],cwd=REPO,check=True);subprocess.run(['git','push','origin',BRANCH],cwd=REPO,check=True,timeout=90)
 print(json.dumps(dict(observed=result['observed_management_utc'],states={k:v['state'] for k,v in candidates.items()},gpu_hours=ledger['consumed_gpu_hours'])),flush=True)
 return result

def main():
 lock=(ROOT/'publisher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);deadline=time.monotonic()+24*3600
 while True:
  try:
   r=observe()
   if r['decisions'] is not None:break
  except (OSError,subprocess.SubprocessError,json.JSONDecodeError) as error:print(json.dumps(dict(publication_error=repr(error),training_modified=False)),flush=True)
  if time.monotonic()>deadline:break
  time.sleep(600)
if __name__=='__main__':main()
