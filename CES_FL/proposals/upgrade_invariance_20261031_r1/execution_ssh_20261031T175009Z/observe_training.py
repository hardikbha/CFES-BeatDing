"""Read-only finite observer for already launched direct-SSH arms. Never launches jobs."""
import datetime, importlib.util, json, os, shlex, subprocess, time, re, fcntl
from pathlib import Path
ROOT=Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_ssh_bounded_20261031T175009Z')
SOURCE=Path('/home/sachin.chaudhary/ces_fl_sources/upgrade_875bce025084871438ef799ca9086985c4375672')
ARMS=['R0_frozen_bce','R1_partial_bce','R2_partial_dro','RN_partial_dro_nochange_pilot','RD_partial_dro_age_pilot']
helper=Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_recovery_20261031T165803Z/runtime/execute_handoff.py')
spec=importlib.util.spec_from_file_location('verified_receipt_helpers',helper);U=importlib.util.module_from_spec(spec);spec.loader.exec_module(U)
U.ROOT=ROOT;U.PUBLIC_BASE='CES_FL/results/'+ROOT.name;U.PUBLIC_STATUS='CES_FL/status/'+ROOT.name+'_live.md'
def active():
    result={}
    for host in ['gpu01','gpu02']:
        code="import os,json,subprocess;from pathlib import Path;out=[];lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines();\nfor line in lines:\n uuid,pid,memory=[x.strip() for x in line.split(',')];\n try:cmd=(Path('/proc')/pid/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()\n except FileNotFoundError:continue\n if "+repr(ROOT.name)+" in cmd and 'train-config' in cmd:out.append({'model_pid':int(pid),'gpu_uuid':uuid,'used_mib':memory,'command':cmd})\nprint(json.dumps(out))"
        p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'/home/sachin.chaudhary/.conda/envs/aiface/bin/python -c '+shlex.quote(code)],capture_output=True,text=True)
        result[host]=json.loads(p.stdout) if p.returncode==0 else {'query_error':p.stderr[-1000:]}
    return result
def observe():
    processes=active();rows={}
    for arm in ARMS:
        submission=U.read(ROOT/'runtime'/f'{arm}.submission.json')
        binding=U.read(ROOT/'runtime'/f'{arm}.binding.json')
        if binding is not None:binding={**binding,'preexisting_compute_processes_present':bool(binding.get('preexisting_compute_processes',0))}
        if binding:binding.pop('preexisting_compute_processes',None)
        records=U.history(ROOT/arm/'training/history.jsonl')
        pro=U.read(ROOT/arm/'training/provenance.json')
        exit=U.read(ROOT/'runtime'/f'{arm}.exit.json');failure=U.read(ROOT/'runtime'/f'{arm}.failure.json')
        entries=processes.get(submission['host'],[]) if submission else []
        live=[p for p in entries if ('/'+arm+'/training') in p['command']] if isinstance(entries,list) else []
        state='TRAINING' if live else ('STARTING_OR_UNVERIFIED' if not exit else 'FINISHED')
        if binding and binding.get('state')=='WAITING_FOR_VRAM_HEADROOM':state='WAITING_FOR_VRAM_HEADROOM'
        if failure or (exit and exit['exit_code']!=0):state='FAILED'
        row={'state':state,'submission':submission,'binding':binding,'model_processes':live,'exit':exit,'failure':failure,
            'completed_epochs':len(records),'history':records,'training_log':str(ROOT/arm/'training.log'),'launcher_log':str(ROOT/'logs'/f'{arm}.ssh.log'),
            'provenance':pro,'cuda_smoke_pass':(ROOT/arm/'CUDA_SMOKE_PASS').exists()}
        if records:row['best_observed_validation']=max(records,key=lambda r:r['validation']['overall']['auc'])
        usage=U.read(ROOT/arm/'training/pair_usage.json')
        if usage:
            draws=list(usage.get('source_draws',{}).values());variants=list(usage.get('scheduled_age_pair_draws',{}).values())
            row['pair_exposure']={'mode':usage['consistency_mode'],'unique_sources':len(draws),'total_source_draws':sum(draws),'min_source_reuse':min(draws) if draws else 0,'max_source_reuse':max(draws) if draws else 0,'scheduled_variants':len(variants),'total_variant_draws':sum(variants)}
        if (ROOT/arm/'training/DONE').exists():
            try:row['checkpoint_review']=U.verify_checkpoint(arm,records);row['state']='COMPLETED_CHECKPOINT_REVIEWED'
            except Exception as e:row['state']='COMPLETED_REVIEW_FAILED';row['review_error']=repr(e)
        rows[arm]=row
    return {'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_mode':'direct_ssh_bounded_shared_gpu','pbs_allocation':None,
        'authorization':U.read(ROOT/'SSH_EXECUTION_AUTHORIZATION.json'),'cuda_gate':U.read(ROOT/'CUDA_GATE_PASSED.json'),'arms':rows,
        'initial_cuda_logs':{h:(ROOT/('SMOKE_'+h)/'smoke_cuda.log').read_text() for h in ['gpu01','gpu02']},
        'source_and_configuration_hashes':U.read(ROOT/'handoff_verification.json'),'source':str(SOURCE),'run_root':str(ROOT),
        'test_evaluated':False,'memory_fraction_per_process':0.1,'minimum_free_mib_at_binding':24576}
def render(d):
    lines=['# CES-FL controlled upgrade — direct SSH training','',
        'Observed UTC: **'+d['observed_utc']+'**. Snapshot from actual GPU processes and saved artifacts.','',
        '**CUDA smoke passed on gpu01 and gpu02: eight tests per node.** Each production arm also repeats input preflight and CUDA smoke.','',
        'Following the latest direct-SSH instruction, these processes have **no PBS allocation or PBS job ID**. The duplicate queued smoke44906 was cancelled before execution; existing jobs44861/44881 were not stopped.','',
        '| Arm | Host | GPU index | Model PID | State | Epochs complete | Best validation AUC (%) |',
        '|---|---|---:|---:|---|---:|---:|']
    for arm,row in d['arms'].items():
        b=row['binding'] or {};sub=row['submission'] or {};live=row['model_processes']
        auc=row.get('best_observed_validation',{}).get('validation',{}).get('overall',{}).get('auc')
        value='—' if auc is None else f'{100*auc:.4f}'
        lines.append(f'| {arm} | {sub.get("host","—")} | {b.get("physical_index","—")} | {live[0]["model_pid"] if live else "—"} | {row["state"]} | {row["completed_epochs"]}/10 | {value} |')
    lines +=['','[Aggregate evidence, logs, hashes, validation subgroup metrics and pair exposure](../../'+U.PUBLIC_BASE+'/aggregate.json).','',
        'The exact approved source remains commit875bce025084871438ef799ca9086985c4375672 from [the supervisor fork](https://github.com/hardiksharma6555/CFES-BeatDing/tree/875bce025084871438ef799ca9086985c4375672). Scientific configuration/input hashes, seed17,10epochs, original sampling, validation-AUC selection and validation-only FPR calibration remain fixed.','',
        'Operational change: bounded GPU sharing via SSH, at least24GiB free at binding, shared project leases, and a0.1PyTorch allocator fraction (about8GiB on80GB H100) through a separately recorded sitecustomize hook. No existing workload is evicted. The memory cap affects execution resources; it does not change the model or loss. Shared GPU contention affects elapsed time.','',
        'RN/RD use six reviewed sources only. They are small-pool diagnostics, not a300-source study or proof of fairness. Test sets are not evaluated or tuned against. The historical CESel-MA gate remains FAIL.','',
        'The finite24h observer only reads/publishes these already launched processes. It cannot launch, retry or cancel jobs. Read observation times. Checkpoint review checks strict reload and recorded selection/calibration, not an independent validation replay.','',
        'Output root: '+str(ROOT), 'Model source: '+str(SOURCE)]
    return '\n'.join(lines)+'\n'
def main():
    guard=(ROOT/'observer.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    U.write(ROOT/'observer.json',{'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only finite24h observer; no launches or retries'})
    start=time.monotonic();last=0;previous=None
    while time.monotonic()-start<24*3600:
        d=observe();U.write(ROOT/'CURRENT.json',d)
        signature=json.dumps({a:{'state':r['state'],'epochs':r['completed_epochs']} for a,r in d['arms'].items()},sort_keys=True)
        if signature!=previous or time.monotonic()-last>1800:
            try:
                body=json.dumps(d,indent=2,allow_nan=False)+'\n';(ROOT/'AGGREGATE.json').write_text(body);(ROOT/'STATUS.md').write_text(render(d))
                c1=U.github_file(U.PUBLIC_BASE+'/aggregate.json',body);c2=U.github_file(U.PUBLIC_STATUS,render(d))
                U.write(ROOT/'last_publication.json',{'aggregate_commit':c1,'status_commit':c2,'utc':d['observed_utc'],'status_path':U.PUBLIC_STATUS})
                print('GITHUB_PROGRESS_PUSH '+c1+' '+c2,flush=True);last=time.monotonic();previous=signature
            except Exception as e:print('PUBLICATION_ERROR '+repr(e),flush=True)
        if all(r['state'] in ['FAILED','COMPLETED_CHECKPOINT_REVIEWED','COMPLETED_REVIEW_FAILED'] for r in d['arms'].values()):return
        time.sleep(30)
if __name__=='__main__':main()
