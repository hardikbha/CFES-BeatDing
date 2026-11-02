"""Read-only finite observation/publication; cannot launch or change experiments."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parent
REPO=Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-forensic-response-publish-c332fd34')
REL='CES_FL/results/forensic_response_recovery_c332fd34_20261103'
DEST=REPO/REL
BRANCH='execution/forensic-response-c332fd34-20261103'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')


def observe():
    ledger=read(ROOT/'budget_ledger.json');live={}
    for host in ('gpu01','gpu02'):
        pids=[s['child_pid'] for s in ledger['gpu_sessions'] if s.get('host')==host and s['state']=='RUNNING' and s.get('child_pid')]
        if not pids:continue
        code='from pathlib import Path;import json; ids='+repr(pids)+';print(json.dumps({str(p):dict(exists=Path("/proc/"+str(p)).exists(),cmdline=Path("/proc/"+str(p)+"/cmdline").read_bytes().replace(bytes([0]),b" ").decode(errors="replace") if Path("/proc/"+str(p)+"/cmdline").exists() else "") for p in ids}))'
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=30)
            live[host]=json.loads(result.stdout) if result.returncode==0 else {'verification':'SSH_FAILED'}
        except subprocess.TimeoutExpired:live[host]={'verification':'SSH_TIMEOUT'}
    studies={}
    for ds in ('ffpp','ff2'):
        run=Path('/home/sachin.chaudhary/ces_fl_runs')/f'forensic_response_overlapfix_20261103_r1_{ds}_s17'
        config=read(run/'config.json');pre=run/'preflight/PREFLIGHT.json'
        data=dict(root=str(run),config_file_sha256=sha(run/'config.json'),config_digest=hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest(),
                  preflight=read(pre) if pre.exists() else None,qualification_recorded=(run/'qualification.json').exists(),arms={})
        if (run/'qualification.json').exists():
            q=read(run/'qualification.json')
            data['qualification_summary']={k:q[k] for k in ('status','source_sha256','config_digest','case_count','real_forgery_validity_proven','scope')}
            data['qualification_sha256']=sha(run/'qualification.json')
            summary=run/'operator_preview/REVIEW_SUMMARY.json'
            if summary.exists():data['operator_review_scope']=read(summary)
        for arm in 'ABCD':
            out=run/arm;records=read(out/'history.json') if (out/'history.json').exists() else []
            result=read(out/'RUN_VERIFIED.json') if (out/'RUN_VERIFIED.json').exists() else None
            best=max(records,key=lambda r:r['validation']['overall']['auc']) if records else None
            sessions=[s for s in ledger['gpu_sessions'] if (s['id'].startswith(ds+'_train_'+arm) or s['id'].startswith(ds+'_'+arm+'_'))]
            current=[s for s in sessions if s['state']=='RUNNING']
            state='PENDING'
            if current:
                s=current[-1];process=live.get(s['host'],{}).get(str(s.get('child_pid')),{});
                state='RUNNING' if process.get('exists') and 'ces_fl_forensic_response.py' in process.get('cmdline','') else 'RUNNING_UNVERIFIED'
            elif result and (out/'DONE').exists():
                if (result['source_sha256']!=ledger['source_sha256'] or result['config_digest']!=data['config_digest'] or result['epochs_completed']!=10 or result['selected_checkpoint_sha256']!=sha(out/'best.pt')):
                    state='COMPLETION_PROVENANCE_MISMATCH'
                else:state='COMPLETED_VERIFIED'
            elif sessions and any(s['state']=='FAILED' for s in sessions):state='FAILED'
            elif (out/'STATUS.json').exists():state=read(out/'STATUS.json')['status']
            data['arms'][arm]=dict(state=state,completed_epochs=len(records),planned_epochs=10,
                    latest_validation=records[-1]['validation'] if records else None,
                    selected_so_far_epoch=best['epoch'] if best else None,
                    selected_so_far_validation=best['validation'] if best else None,
                    completed_verification=result,
                    selected_checkpoint_sha256=sha(out/'best.pt') if (out/'best.pt').exists() else None,
                    coverage=[{k:r[k] for k in ('epoch','unique_originals','originals_once_each','original_real_pairs','epoch_seconds','auxiliary_schedule')} for r in records],
                    sessions=sessions)
        studies[ds]=data
    snap=dict(observed_management_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit='c332fd34fb603b8919b15fee08e8d022fb9f57c8',source_sha256=ledger['source_sha256'],
              budget=ledger,actual_process_observations=live,datasets=studies,
              coordinator=read(ROOT/'COORDINATOR_STATUS.json') if (ROOT/'COORDINATOR_STATUS.json').exists() else None,
              test_evaluation=False,pbs_job_ids_for_this_study=[],route='User-authorized direct SSH with explicit shared pool-v2 GPU leases; not PBS allocations',
              review_scope='Named AI visual operator review; no human approval or real-forgery validity fabricated. Private reviews/images not published.')
    write(DEST/'LIVE_STATE.json',snap)
    lines=['# Corrected-source forensic-response study: actual progress','',f"Observed management UTC: {snap['observed_management_utc']}",'',
      'Execution commit `c332fd34fb603b8919b15fee08e8d022fb9f57c8`; script SHA256 `89530025a00b9b0405fba7026d3eed5f2749d86620174ad3ffde93a44f8b770b`.', '',
      snap['route']+'. Existing Ding source/jobs preserved. No TEST inference.','',
      f"New-study task-held GPU time: **{ledger['consumed_gpu_hours']:.4f}/24TOTAL GPU-hours**. Unspent conservative reservations: {ledger['reserved_gpu_hours']:.4f}hours. Count both GPUs at full task-held wall time, even when shared; not kernel-only/exclusive-H100 utilization. CPU preflight/review are not GPU time.",'',
      '| Dataset | Arm | Verified process/completion state | Complete epochs | Selected so far | VAL AUC (%) | gender×race F_FPR (fraction) | FPR gap (pp) | Worst AUC (%) |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for ds,data in studies.items():
        for arm,row in data['arms'].items():
            v=row['selected_so_far_validation'];g=v['axes']['gender_race'] if v else None
            values=[f"{100*v['overall']['auc']:.4f}",f"{g['F_FPR']:.6f}",f"{100*g['fpr_gap']:.4f}",f"{100*g['worst_group_auc']:.4f}"] if v else ['—']*4
            lines.append(f"| {ds} | {arm} | {row['state']} | {row['completed_epochs']}/10 | {row['selected_so_far_epoch'] or '—'} | "+' | '.join(values)+' |')
    lines+=['','All values above are VAL. Selected-so-far values remain preliminary until ten epochs and native independent replay pass. No fairness/model/threshold selection against TEST. Single-seed results do not establish statistical superiority.','',
      'Full per-axis/group supports, FPR/FNR, calibration, coverage, exact configuration hashes, GPU leases/PIDs, checkpoint hashes and errors are in LIVE_STATE.json. Age/skin labels are inherited proxies; recorded contributor IDs do not certify biological identities.','',
      'FF++ reviewed88 procedural cases across22 originals/18recorded identity-source IDs. The review is an explicitly named AI visual screen, not human annotation, per-view universal approval or proof of actual forgery validity. Source-bound qualification receipt stays private.','',
      'The original six-key audit is unchanged; two TRAIN/VAL declarations checked, four TEST-related declarations DEFERRED. Known163TRAIN/VAL contributor overlaps remain disclosed.','',
      'The requested hardiksharma6555 fork rejected the evidence push. This separate branch is published in the previously authorized hardikbha upstream; no automatic merge.','',
      'Finite observer: every10minutes for at most48hours, with transient SSH/push errors reported/retried. It cannot launch/retry training, generate reviews or access TEST. Separate explicit study_coordinator.py handles only the already-authorized A-D sequence and hash-matched documented budget resumes.','']
    (DEST/'STATUS.md').write_text('\n'.join(lines))
    allowed=[REL+'/LIVE_STATE.json',REL+'/STATUS.md']
    subprocess.run(['git','add','--',*allowed],cwd=REPO,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).splitlines()
    if not set(staged)<=set(allowed):raise RuntimeError('Unexpected staged files: preserve other work')
    subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
    if staged:
        subprocess.run(['git','-c','gc.auto=0','commit','-m','Publish actual forensic-response stages, validation and GPU-hour ledger'],cwd=REPO,check=True)
        subprocess.run(['git','push','origin',BRANCH],cwd=REPO,check=True,timeout=90)
    print(json.dumps(dict(observed=snap['observed_management_utc'],gpu_hours=ledger['consumed_gpu_hours'],states={d:{a:r['state'] for a,r in data['arms'].items()} for d,data in studies.items()})),flush=True)
    return snap


def main():
    lock=(ROOT/'publisher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    until=time.monotonic()+48*3600
    while True:
        try:
            snap=observe()
            if all(r['state']=='COMPLETED_VERIFIED' for d in snap['datasets'].values() for r in d['arms'].values()):break
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError,json.JSONDecodeError) as e:
            print(json.dumps(dict(publication_error=repr(e),training_modified=False)),flush=True)
        if time.monotonic()>until:break
        time.sleep(600)


if __name__=='__main__':main()
