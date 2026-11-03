"""Finite aggregate-only publication. No training, scheduling or review capability."""
import datetime,fcntl,hashlib,json,shlex,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/hierarchical_real_risk_s17_20261103_r1')
REPO=Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-hier-real-prep-20261103')
REL='CES_FL/results/hierarchical_real_risk_s17_20261103_r1'
BRANCH='experiment/cesfl-hier-real-s17-20261103'
SHA='2549cb1345d1e2ee617e8a77ca288a0acbbfd05cf480f7b91a46b8627fcb74b2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def observe():
    ledger=read('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/budget_ledger.json');sessions=[s for s in ledger['gpu_sessions'] if s['id'].startswith('hier_')]
    active=[s for s in sessions if s['id']=='hier_H_joint_real' and s['state']=='RUNNING'];actual={}
    for s in active:
        pid=s.get('child_pid')
        if not pid:continue
        code='from pathlib import Path;p=Path("/proc/'+str(pid)+'/cmdline");print(p.read_bytes().replace(bytes([0]),b" ").decode() if p.exists() else "")'
        p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',s['host'],'python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=25)
        actual[str(pid)]=p.stdout.strip() if p.returncode==0 else 'SSH_FAILED_UNVERIFIED'
    output=ROOT/'H_joint_real/training';hist=read(output/'history.json') if (output/'history.json').exists() else []
    selected=max(hist,key=lambda x:x['validation']['overall']['auc']) if hist else None
    receipt=read(output/'RUN_VERIFIED.json') if (output/'RUN_VERIFIED.json').exists() else None
    state='CHECKS_OR_QUALIFICATION_PENDING'
    if active:state='RUNNING' if any('ces_fl_forensic_response.py' in x and str(ROOT/'H_joint_real/config.json') in x for x in actual.values()) else 'RUNNING_UNVERIFIED'
    elif receipt and (output/'DONE').exists():
        state='COMPLETED_VERIFIED' if receipt['source_sha256']==SHA and receipt['epochs_completed']==10 and receipt['selected_checkpoint_sha256']==sha(output/'best.pt') and not receipt['test_evaluated'] else 'PROVENANCE_MISMATCH'
    elif any(s['id']=='hier_H_joint_real' and s['state']=='FAILED' for s in sessions):state='FAILED_OR_INCOMPLETE'
    decision=read(ROOT/'validation_analysis/DECISIONS.json') if (ROOT/'validation_analysis/DECISIONS.json').exists() else None
    d=read('/home/sachin.chaudhary/ces_fl_runs/forensic_response_overlapfix_20261103_r1_ffpp_s17/D/RUN_VERIFIED.json')
    result=dict(observed_management_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state=state,scientific_commit='3f5f3c16839bf4b96209bea3f968255b5f351ebb',source_sha256=SHA,
                config_sha256=sha(ROOT/'H_joint_real/config.json'),epochs_completed=len(hist),selected_epoch=selected['epoch'] if selected else None,
                selected_validation=selected['validation'] if selected else None,native_completion=receipt,
                exposure=[{k:e.get(k) for k in ['epoch','steps','unique_originals','originals_once_each','original_real_pairs','original_real_pairs_by_axis_group','auxiliary_original_draws_by_axis_group_label','alignment_multiplier','epoch_seconds']} for e in hist],
                actual_process_observations=actual,gpu_sessions=sessions,total_gpu_hours=ledger['consumed_gpu_hours'],remaining_gpu_hours=24-ledger['consumed_gpu_hours'],
                decision=decision,posthoc=read(ROOT/'POSTHOC_STATUS.json') if (ROOT/'POSTHOC_STATUS.json').exists() else None,
                reference_D=d,structural_uncertainty=read(ROOT/'STRUCTURAL_BOOTSTRAP_LIMIT.json'),no_new_ding_computation=True,no_FF2=True,test_read=False,
                scope='One H config,seed17,FF++only; explicitly authorized pool-v2 UUID route, no PBS/exclusive GPU allocation claimed')
    dest=REPO/REL;dest.mkdir(exist_ok=True)
    (dest/'LIVE_STATE.json').write_text(json.dumps(result,indent=2)+'\n')
    v=result['selected_validation'];lines=['# CES-FL hierarchical original-real risk','',result['observed_management_utc']+' management UTC.','',
       f'**H: {state}; {len(hist)}/10 epochs completed.** '+('Final decision: **'+decision['decision']+'**.' if decision else 'No final decision yet.'),'',
       'One seed17 candidate. No new Ding-model computation, FairForge-2 experiment or TEST inference. Existing jobs/results preserved.','',
       f'Original shared budget{ledger["consumed_gpu_hours"]:.4f}/24GPU-hours; remaining{24-ledger["consumed_gpu_hours"]:.4f}h. New-study cap4.0h; all held time charged.','',
       '|Arm|Selected epoch|VAL AUC %|Gender F_FPR fraction|Intersection F_FPR fraction|Age F_FPR fraction|','|---|---:|---:|---:|---:|---:|']
    for label,epoch,vv in [('D',d['selected_epoch'],d['validation']),('H',result['selected_epoch'],v)]:
        if vv:lines.append(f'|{label}|{epoch}|{100*vv["overall"]["auc"]:.6f}|{vv["axes"]["gender"]["F_FPR"]:.6f}|{vv["axes"]["gender_race"]["F_FPR"]:.6f}|{vv["axes"]["age_bin"]["F_FPR"]:.6f}|')
    lines+=['','H values are selected-so-far until native completion. All axes, group support, FPR/FNR and actual exposure are in LIVE_STATE.json. E/F already completed and rejected; they are not rerun.','',
            'Exact proposal/config/guards: ../../proposals/hierarchical_real_risk_s17_20261103_r1/README.md. Gender target≤0.042798380714166834; primary intersection<0.3793453731153111; all prior guards retained. Fixed highest-VAL-AUC selection. No fairness checkpoint switch.','',
            'The fixed component bootstrap has746/1000usable primary draws from labels alone, below the required900. PASS is structurally unavailable under the retained rule; even numerical success remains INCONCLUSIVE. The single experiment measures the prespecified numerical effect and cannot establish a paper/Ding win.','',
            'Gates are learned weights; maps are signed local contributions. Visual recipe qualification is not causal/real-forgery validity or human demographic ground truth. Face images, reviews, row scores and checkpoints stay private. Publisher cannot launch jobs or create approvals; finite24h observation, ten-minute cadence.']
    (dest/'STATUS.md').write_text('\n'.join(lines)+'\n')
    allowed=[REL+'/LIVE_STATE.json',REL+'/STATUS.md']
    if decision:
        for name in ['DECISIONS.json','H_vs_D.json','REPORT.md']:
            src=ROOT/'validation_analysis'/name
            if src.exists():shutil.copy2(src,dest/name);allowed.append(REL+'/'+name)
        hd=dest/'heatmaps';hd.mkdir(exist_ok=True)
        for src in (ROOT/'validation_analysis/heatmaps').glob('*_rates.png'):shutil.copy2(src,hd/src.name);allowed.append(REL+'/heatmaps/'+src.name)
    subprocess.run(['git','add','--',*allowed],cwd=REPO,check=True)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).splitlines()
    if not set(staged)<=set(allowed):raise RuntimeError('Unexpected staged work; preserve')
    if staged:
        subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True);subprocess.run(['git','-c','gc.auto=0','commit','-m','Publish actual one-candidate CES-FL validation progress and fixed-guard outcome'],cwd=REPO,check=True);subprocess.run(['git','push','origin',BRANCH],cwd=REPO,check=True,timeout=90)
    print(json.dumps(dict(observed=result['observed_management_utc'],state=state,epochs=len(hist),decision=decision['decision'] if decision else None)),flush=True)
    return decision
def main():
    lock=(ROOT/'publisher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        try:
            if observe():return
        except (OSError,subprocess.SubprocessError,json.JSONDecodeError) as e:print(json.dumps(dict(publication_error=repr(e),training_modified=False)),flush=True)
        time.sleep(600)
if __name__=='__main__':main()
