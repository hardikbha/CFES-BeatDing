"""Manual safe aggregate snapshot; never submits, releases, retries or terminates jobs."""
import datetime,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/lh_sham_noalign_s17_20261103T165600Z')
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'CES_FL/results/lh_sham_noalign_s17_20261103T165600Z'
OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
data=dict(observed_management_utc=now,execution_commit='35ffbd21cc6a033f9b9bf4cb1f92f5e417debe8d',source_sha256='1b6dc1b0e6d2b2220da897500a4d1bffb7764280df642abf6befdd1cf511ed01',arms={})
lines=['# SHAM-FIX and NO-ALIGN: observed status','',now,'','Source execution commit: `'+data['execution_commit']+'`. Independent three-epoch ablations against completed lower-LR L_H; seed17, FF++ TRAIN/VAL only. No TEST.','', '|Arm|PBS job|Observed scheduler state|Completed epochs|Last recorded VAL AUC %|','|---|---|---|---:|---:|']
for arm in ('SHAM_FIX','NO_ALIGN'):
 r=ROOT/arm;dest=OUT/arm;dest.mkdir(exist_ok=True);sub=read(r/'SUBMISSION.json');job=sub['stdout'].strip()
 q=subprocess.run(['qstat','-f','-F','json',job],capture_output=True,text=True)
 state='UNVERIFIED';scheduler=None
 if not q.returncode:
  full=json.loads(q.stdout)['Jobs'][job];state=full['job_state']
  scheduler={k:full[k] for k in ('job_state','queue','Resource_List','resources_used','exec_host','exec_vnode','stime','comment') if k in full}
 h=read(r/'training/history.json') if (r/'training/history.json').exists() else []
 entry=dict(job_id=job,scheduler_state=state,scheduler_observation=scheduler,completed_epochs=len(h),history=h,paths=dict(root=str(r),training_log=str(r/'train.log')))
 for name in ('REQUEST_CHECK.json','SOURCE_CHECKOUT.json','SUBMISSION.json','ALLOCATION.json','ALLOCATION_END.json','SCHEDULER_ACCOUNTING.json','preflight/PREFLIGHT.json','review/qualification.json','smoke_cuda/SMOKE_RESULT.json','dry/DRY_RUN.json','training/RUN_VERIFIED.json','training/history.json','comparison/COMPARISON.json','comparison/REPORT.md','comparison/ALL_FAIRNESS_METRICS.csv','comparison/GROUP_COUNTS_AND_RATES.csv','loss_tests_cuda.log'):
  src=r/name
  if src.exists():
   dst=dest/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 entry['receipts']={n:str(r/n) for n in ('preflight/PREFLIGHT.json','review/qualification.json','smoke_cuda/SMOKE_RESULT.json','dry/DRY_RUN.json','training/RUN_VERIFIED.json') if (r/n).exists()}
 data['arms'][arm]=entry
 auc=f"{100*h[-1]['validation']['overall']['auc']:.6f}" if h else '—'
 lines.append(f'|{arm}|{job}|{state}|{len(h)}/3|{auc}|')
 if h:
  lines+=['','### '+arm+' recorded epochs','','|Epoch|AUC %|FNR %|Gender F_FPR|Race|Gender×race|Age|Skin|ids8|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for e in h:
   v=e['validation'];lines.append('|'+str(e['epoch'])+'|'+f"{100*v['overall']['auc']:.6f}|{100*v['overall']['fnr']:.6f}|"+'|'.join(f"{v['axes'][a]['F_FPR']:.6f}" for a in ('gender','race','gender_race','age_bin','skin_bin','ids8'))+'|')
for name in ('head_tests_cpu.log','loss_tests_cpu.log'):
 if (ROOT/name).exists():shutil.copyfile(ROOT/name,OUT/name)
ledger=read(Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/budget_ledger.json'))
data['budget']={k:ledger[k] for k in ('initial_total_gpu_hours','consumed_gpu_hours','reserved_gpu_hours')}
data['sessions']=[s for s in ledger['gpu_sessions'] if s['id'].startswith('lh_sham_noalign_s17_20261103')]
lines+=['','Completed epochs are recorded history, not proof of a live process. H=held, Q=queued, R=running allocation; GPU checks precede training. Qualification is reused only on exact matching bytes and retains the original AI procedural-review scope. Both runs start from P50 plus the identical fresh head; neither starts from a trained L_H checkpoint.','', 'Budget: '+json.dumps(data['budget'])+'. A maximum 3 GPU-hours is reserved across the two arms.','', 'All fairness values are fractions. Keep regressions visible. Single-seed validation findings are not a Ding TEST comparison or proof of generalization. This helper publishes only aggregate evidence and does not control execution.']
(OUT/'STATUS.json').write_text(json.dumps(data,indent=2)+'\n');(OUT/'STATUS.md').write_text('\n'.join(lines)+'\n')
print(OUT)
