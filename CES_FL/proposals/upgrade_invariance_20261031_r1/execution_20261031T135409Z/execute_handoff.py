"""Finite executor for the five exact user-authorized upgrade submissions.

No retries, test evaluation, generation, configuration changes or cancellations.
CUDA smoke must actually pass before any training arm is submitted.
"""
import base64, datetime, fcntl, hashlib, json, os, re, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_20261031T135409Z')
SOURCE=Path('/home/sachin.chaudhary/ces_fl_sources/upgrade_875bce025084871438ef799ca9086985c4375672')
COMMIT='875bce025084871438ef799ca9086985c4375672'
ARMS=['R0_frozen_bce','R1_partial_bce','R2_partial_dro','RN_partial_dro_nochange_pilot','RD_partial_dro_age_pilot']
REPOSITORY='hardikbha/CFES-BeatDing'
PUBLIC_BASE='CES_FL/results/'+ROOT.name
PUBLIC_STATUS='CES_FL/status/'+ROOT.name+'_live.md'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path,default=None):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return default
def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def command(args,**kwargs):return subprocess.run(args,capture_output=True,text=True,**kwargs)
def scheduler(job):
    p=command(['qstat','-xf','-F','json',job])
    if p.returncode:return {'job_state':'UNVERIFIED','query_error':p.stderr.strip()}
    d=json.loads(p.stdout)['Jobs'][job]
    return {k:d[k] for k in ['Job_Name','job_state','queue','Exit_status','exec_host','exec_vnode','Resource_List','resources_used','comment'] if k in d}
def pinned():
    expected=read(ROOT/'handoff_verification.json')
    assert expected['git_commit']==COMMIT and expected['detached'] and expected['source_clean']
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert command(['git','-C',str(SOURCE),'symbolic-ref','-q','HEAD']).returncode==1
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--','CES_FL'],text=True).strip()
    for name,digest in expected['sha256'].items():
        assert sha(SOURCE/name)==digest,'Changed pinned source/config: '+name
    checks=read(ROOT/'preflight_summary.json')
    assert len(checks)==5 and {r['arm'] for r in checks}==set(ARMS)
    assert all(r['exit_code']==0 for r in checks)
def smoke_passed():
    receipt=read(ROOT/'smoke_submission.json');state=scheduler(receipt['job_id'])
    write(ROOT/'runtime/SMOKE_R1.scheduler.json',state)
    exit_receipt=read(ROOT/'runtime/SMOKE_R1.exit.json')
    if state.get('job_state')=='F' and state.get('Exit_status')!=0:
        raise RuntimeError('CUDA smoke PBS job failed: '+json.dumps(state))
    if exit_receipt and exit_receipt['exit_code']!=0:
        raise RuntimeError('CUDA smoke launcher failed: '+json.dumps(exit_receipt))
    if state.get('job_state')!='F':return False
    assert state['Exit_status']==0 and exit_receipt and exit_receipt['exit_code']==0
    out=ROOT/'SMOKE_R1';assert (out/'CUDA_SMOKE_PASS').is_file(),'Missing CUDA_SMOKE_PASS'
    log=(out/'smoke_cuda.log').read_text()
    assert re.search(r'Ran 8 tests in ',log) and re.search(r'\nOK\s*$',log),'CUDA test summary missing'
    launch=read(out/'launch.json')
    assert launch['git_commit']==COMMIT and launch['smoke_only'] is True
    assert launch['pbs_jobid']==receipt['job_id']
    assert json.loads((out/'preflight.log').read_text().strip().splitlines()[-1])['status']=='PASS'
    binding=read(ROOT/'runtime/SMOKE_R1.binding.json')
    assert binding['state']=='BOUND' and binding['gpu_uuid']==launch['cuda_visible_devices']
    write(ROOT/'CUDA_GATE_PASSED.json',{'observed_utc':now(),'pbs_job_id':receipt['job_id'],'pbs':state,
        'cuda_binding':binding,'cuda_smoke_sha256':sha(out/'smoke_cuda.log'),'marker':str(out/'CUDA_SMOKE_PASS')})
    return True
def submit(arm):
    assert arm in ARMS
    pinned()
    gate=read(ROOT/'CUDA_GATE_PASSED.json')
    assert gate and gate['pbs_job_id']==read(ROOT/'smoke_submission.json')['job_id'],'CUDA gate not passed'
    path=ROOT/'runtime'/f'{arm}.submission.json'
    if path.exists():return read(path)
    attempt=ROOT/'runtime'/f'{arm}.attempt.json'
    if attempt.exists():raise RuntimeError('Prior submission attempt requires manual reconciliation: '+arm)
    alljobs=command(['qstat','-f','-F','json'])
    assert alljobs.returncode==0,'Cannot verify duplicate jobs'
    for jid,data in json.loads(alljobs.stdout).get('Jobs',{}).items():
        variables=data.get('Variable_List',{})
        expected_config=f'CES_FL/proposals/upgrade_invariance_20261031_r1/configs/{arm}.json'
        if isinstance(variables,dict) and variables.get('CES_SOURCE_DIR')==str(SOURCE) and (variables.get('CES_UPGRADE_STAGE')==arm or (variables.get('CES_TRAIN_CONFIG')==expected_config and variables.get('CES_SMOKE_ONLY')=='0')):
            raise RuntimeError('Existing arm job; not resubmitting: '+arm+' '+jid)
    config=f'CES_FL/proposals/upgrade_invariance_20261031_r1/configs/{arm}.json'
    output=ROOT/arm;assert not output.exists()
    variables={'CES_SOURCE_DIR':str(SOURCE),'CES_TRAIN_CONFIG':config,'CES_OUTPUT_DIR':str(output),'CES_SMOKE_ONLY':'0','CES_UPGRADE_STAGE':arm}
    cmd=['qsub','-N','cesup_'+arm.split('_')[0],'-o',str(ROOT/'logs'/f'{arm}.pbs.log'),'-v',
        ','.join(k+'='+v for k,v in variables.items()),str(ROOT/'runtime/allocated_upgrade.pbs')]
    record={'arm':arm,'command':cmd,'config_sha256':sha(SOURCE/config),'submitted_utc':now(),'output':str(output)}
    write(attempt,record)
    with (ROOT/'runtime'/f'{arm}.qsub.stdout').open('w') as stdout,(ROOT/'runtime'/f'{arm}.qsub.stderr').open('w') as stderr:
        result=subprocess.run(cmd,stdout=stdout,stderr=stderr,text=True)
    record.update(returncode=result.returncode,stdout=(ROOT/'runtime'/f'{arm}.qsub.stdout').read_text(),
                  stderr=(ROOT/'runtime'/f'{arm}.qsub.stderr').read_text())
    if result.returncode:
        write(path,record);raise RuntimeError('qsub failed for '+arm+': '+record['stderr'])
    jid=record['stdout'].strip();assert re.fullmatch(r'\d+\.mgmt01',jid),'Unexpected qsub response; no automatic retry'
    record['job_id']=jid;write(path,record)
    print('TRAINING_JOB_SUBMITTED '+arm+' '+jid,flush=True)
    return record
def history(path):
    if not path.exists():return []
    records=[];lines=path.read_text().splitlines()
    for i,line in enumerate(lines):
        try:records.append(json.loads(line))
        except json.JSONDecodeError:
            if i<len(lines)-1:raise
    return records
def verify_checkpoint(arm,records):
    training=ROOT/arm/'training';target=ROOT/'runtime'/f'{arm}.checkpoint_review.json'
    if target.exists():return read(target)
    import torch
    sys.path.insert(0,str(SOURCE/'CES_FL/src'))
    import ces_fl_upgrade as up
    config=read(SOURCE/f'CES_FL/proposals/upgrade_invariance_20261031_r1/configs/{arm}.json')
    assert len(records)==config['epochs']
    best=max(records,key=lambda r:r['validation']['overall']['auc'])
    model,checkpoint=up.load_model(training/'best.pt','cpu')
    assert checkpoint['config']==config and checkpoint['input_hashes']==config['input_hashes']
    assert checkpoint['epoch']==best['epoch'] and checkpoint['validation']==best['validation']
    assert checkpoint['calibration']==best['calibration'] and checkpoint['synthetic_smoke'] is False
    expected=read(ROOT/'handoff_verification.json')['sha256']
    assert checkpoint['source_sha256']=={'upgrade':expected['CES_FL/src/ces_fl_upgrade.py'],'base':expected['CES_FL/src/ces_fl_diagram_model_v021.py']}
    assert all(torch.isfinite(v).all().item() for v in model.state_dict().values() if v.is_floating_point())
    result={'status':'CHECKPOINT_RELOADED_HISTORY_SELECTION_AND_CALIBRATION_METADATA_MATCH',
        'validation_forward_independently_replayed':False,'selected_epoch':checkpoint['epoch'],
        'best_checkpoint_sha256':sha(training/'best.pt'),'last_checkpoint_sha256':sha(training/'last.pt'),
        'calibration':checkpoint['calibration'],'observed_utc':now(),
        'note':'Strict reload and recorded validation-only selection/calibration checked; no independent validation replay or intervention-validity certification.'}
    write(target,result);del model,checkpoint
    return result
def observe(stopped=None):
    result={'observed_utc':now(),'run_root':str(ROOT),'source':str(SOURCE),'commit':COMMIT,
        'scope':'Five exact Xception configs; RN/RD six-source diagnostics; no test evaluation or generation',
        'smoke':{'submission':read(ROOT/'smoke_submission.json'),'scheduler':read(ROOT/'runtime/SMOKE_R1.scheduler.json'),'gate':read(ROOT/'CUDA_GATE_PASSED.json')},
        'stopped':stopped,'arms':{}}
    for arm in ARMS:
        submission=read(ROOT/'runtime'/f'{arm}.submission.json')
        item={'submission':submission,'state':'NOT_SUBMITTED','training_log':str(ROOT/arm/'training.log'),'pbs_log':str(ROOT/'logs'/f'{arm}.pbs.log')}
        if submission and submission.get('job_id'):
            item['scheduler']=scheduler(submission['job_id']);state=item['scheduler'].get('job_state')
            item['state']={'Q':'QUEUED','H':'HELD','R':'ALLOCATED','F':'FINISHED'}.get(state,'UNVERIFIED')
            item['binding']=read(ROOT/'runtime'/f'{arm}.binding.json')
            if item['binding'] and item['binding']['state']=='WAITING_FOR_EMPTY_GPU':item['state']='WAITING_FOR_EMPTY_GPU'
            rows=history(ROOT/arm/'training/history.jsonl');item['completed_epochs']=len(rows)
            if rows:
                item['history']=rows;item['last_epoch']=rows[-1]['epoch']
                item['best_observed_validation']=max(rows,key=lambda x:x['validation']['overall']['auc'])
                if state=='R':item['state']='TRAINING_RECORDED'
            item['failure']=read(ROOT/'runtime'/f'{arm}.failure.json')
            if state=='F' and item['scheduler'].get('Exit_status')!=0:item['state']='FAILED'
            if item['failure']:item['state']='FAILED'
            if (ROOT/arm/'training/DONE').exists():
                try:item['checkpoint_review']=verify_checkpoint(arm,rows);item['state']='COMPLETED_CHECKPOINT_REVIEWED'
                except Exception as error:item['state']='COMPLETED_REVIEW_FAILED';item['review_error']=repr(error)
            usage=read(ROOT/arm/'training/pair_usage.json')
            if usage:
                draws=list(usage.get('source_draws',{}).values());variants=list(usage.get('scheduled_age_pair_draws',{}).values())
                item['pair_exposure']={'consistency_mode':usage['consistency_mode'],'unique_source_count':len(draws),
                    'source_draw_count':sum(draws),'source_reuse_min':min(draws) if draws else 0,
                    'source_reuse_max':max(draws) if draws else 0,'scheduled_variant_count':len(variants),'scheduled_variant_draw_count':sum(variants)}
        result['arms'][arm]=item
    result['input_preflight']=read(ROOT/'preflight_summary.json');result['handoff_verification']=read(ROOT/'handoff_verification.json')
    result['cpu_software_log']=(ROOT/'logs/cpu_software_tests.log').read_text()
    result['cuda_software_log']=(ROOT/'SMOKE_R1/smoke_cuda.log').read_text() if (ROOT/'SMOKE_R1/smoke_cuda.log').exists() else None
    result['input_preflight_outputs']={p.stem:p.read_text() for p in (ROOT/'preflight').glob('*.log')}
    return result
def render(record):
    lines=['# Controlled CES-FL upgrade: actual execution status','',
        'Observed UTC: **'+record['observed_utc']+'**. Queued/allocated is not proof of training.','',
        'Exact source commit: '+COMMIT+' from [the supplied fork](https://github.com/hardiksharma6555/CFES-BeatDing/tree/'+COMMIT+').','',
        'CPU software suite: **19 tests passed**. External-input preflight: **five exact configurations passed**.',
        'TRAIN: 76,139 records; validation: 25,386. The preserved protocol declares 163 overlapping contributor identifiers; this is not an identity-disjoint benchmark.','',
        'Initial smoke job: **'+record['smoke']['submission']['job_id']+'**; CUDA gate: **'+('PASS' if record['smoke']['gate'] else 'PENDING')+'**.','',
        '| Arm | PBS job | Observed status | Epochs recorded | Best observed validation AUC (%) |',
        '|---|---|---|---:|---:|']
    for arm,item in record['arms'].items():
        job=(item['submission'] or {}).get('job_id','—')
        auc=item.get('best_observed_validation',{}).get('validation',{}).get('overall',{}).get('auc')
        value='—' if auc is None else f'{100*auc:.4f}'
        lines.append(f'| {arm} | {job} | {item["state"]} | {item.get("completed_epochs",0)} | {value} |')
    if record['stopped']:lines+=['','**LAUNCH SEQUENCE STOPPED:** '+record['stopped']]
    lines+=['','[Aggregate evidence, validation subgroup metrics, exposure and software logs](../../'+PUBLIC_BASE+'/aggregate.json).','',
        'RN/RD are six-source small-pool diagnostics; repeated draws do not establish broader coverage. No expanded300-source study, new images, forensic loss, map consistency, test tuning or test evaluation is authorized here.','',
        'Selection is validation AUC only. Calibration remains the selected checkpoint’s validation-real empirical FPR budget. Observed metrics do not establish fairness improvement or a win over Ding.','',
        'GPU jobs request ngpus=1. The site exposes an allocated host/count rather than a UUID mapping; the operational wrapper verifies the allocation, binds a verified empty physical UUID under the existing project lock, then invokes the byte-identical approved upgrade_one_gpu.pbs. Allocation/binding evidence is retained.','',
        'The finite handoff executor submits only the five preauthorized arms after successful initial CUDA smoke, never retries failures, and observes/publishes for at most24h. It does not cancel other jobs or modify research code/configuration. Checkpoint review confirms strict reload and matching recorded selection/calibration; it is not an independent validation replay.','',
        'Cluster output root: '+str(ROOT)+'.','Logs: '+str(ROOT/'logs')+'. Source: '+str(SOURCE)+'.']
    return '\n'.join(lines)+'\n'
def github_file(path,body):
    endpoint='repos/'+REPOSITORY+'/contents/'+path;p=command(['gh','api',endpoint])
    if p.returncode:
        if '404' not in p.stderr:raise RuntimeError('GitHub read failed: '+p.stderr[:500])
        current=None
    else:current=json.loads(p.stdout)
    payload={'message':'Record '+ROOT.name+' verified execution observations','content':base64.b64encode(body.encode()).decode(),'branch':'main'}
    if current:payload['sha']=current['sha']
    payload_path=ROOT/'runtime/github_payload.json';write(payload_path,payload)
    p=command(['gh','api','--method','PUT',endpoint,'--input',str(payload_path)])
    if p.returncode:raise RuntimeError('GitHub publish failed: '+p.stderr[:500])
    return json.loads(p.stdout)['commit']['sha']
def publish(record):
    data=json.dumps(record,indent=2,allow_nan=False)+'\n'
    (ROOT/'AGGREGATE.json').write_text(data);(ROOT/'STATUS.md').write_text(render(record))
    first=github_file(PUBLIC_BASE+'/aggregate.json',data);last=github_file(PUBLIC_STATUS,render(record))
    print('GITHUB_PROGRESS_PUSH '+first+' '+last+' '+now(),flush=True)
    write(ROOT/'runtime/last_publication.json',{'aggregate_commit':first,'status_commit':last,'observed_utc':now(),'status_path':PUBLIC_STATUS})
def main():
    guard=(ROOT/'controller.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(ROOT/'controller.json',{'pid':os.getpid(),'started_utc':now(),'scope':'Only five preauthorized submissions after CUDA gate; finite24h'})
    pinned();started=time.monotonic();last_publish=0;last_signature=None;submitted=False;stopped=None
    while time.monotonic()-started<24*3600:
        if not stopped and not submitted:
            try:
                if smoke_passed():
                    for arm in ARMS:submit(arm)
                    submitted=True
            except Exception as error:
                stopped=repr(error);write(ROOT/'LAUNCH_STOPPED.json',{'error':stopped,'time':now(),'traceback':traceback.format_exc()})
                print('LAUNCH_STOPPED '+stopped,flush=True)
        snapshot=observe(stopped);write(ROOT/'CURRENT.json',snapshot)
        signature=json.dumps({'smoke':bool(snapshot['smoke']['gate']),'stopped':stopped,
            'arms':{a:{k:i.get(k) for k in ['state','completed_epochs','review_error']} for a,i in snapshot['arms'].items()}},sort_keys=True)
        if signature!=last_signature or time.monotonic()-last_publish>1800:
            try:publish(snapshot);last_signature=signature;last_publish=time.monotonic()
            except Exception as error:print('PUBLICATION_ERROR '+repr(error),flush=True)
        if stopped:return 1
        if submitted and all(i['state'] in ['FAILED','COMPLETED_CHECKPOINT_REVIEWED','COMPLETED_REVIEW_FAILED'] for i in snapshot['arms'].values()):return 0
        time.sleep(30)
    write(ROOT/'CONTROLLER_TIME_LIMIT.json',{'observed_utc':now(),'note':'24h observation limit reached; scheduler jobs not changed.'})
    return 0
if __name__=='__main__':raise SystemExit(main())
