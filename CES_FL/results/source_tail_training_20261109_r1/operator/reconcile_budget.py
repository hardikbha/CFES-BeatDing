"""Conservatively reconcile recorded historical inference; no model/data access."""
import datetime,fcntl,hashlib,json,os,re
from pathlib import Path

P=Path('/home/sachin.chaudhary/ces_fl_runs/source_tail_training_20261109_r1')
L=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_baseline_suite_s17_20261103T110324Z/additional72_gpu_hours.json')
history=P/'historical_tool_commands_PRIVATE.json'
u=json.loads(history.read_text());events=[];devices=set()
for e in u.values():
    bound=re.findall(r'CUDA_VISIBLE_DEVICES\s*=\s*([0-9]+(?:,[0-9]+)*)',e['command_PRIVATE'])
    if not bound:continue
    assert e.get('end_timestamp') and not e.get('is_error')
    assert not any(s in e['command_PRIVATE'] for s in ['nohup','setsid','disown'])
    devices.update(g for s in bound for g in s.split(','))
    events.append({k:e[k] for k in ['timestamp','end_timestamp','command_sha256']})
assert len(events)==19 and devices=={'2','3','7'}
epoch=lambda t:datetime.datetime.fromisoformat(t.replace('Z','+00:00')).timestamp()
start=min(epoch(e['timestamp']) for e in events);end=max(epoch(e['end_timestamp']) for e in events)
upper=(end-start)*len(devices)/3600
key='ff2_test_sanity_20261108_conservative_reconciled_20261109'
with Path(str(L)+'.lock').open('a') as lock:
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    s=json.loads(L.read_text());assert s['cap_gpu_hours']==72 and s['max_concurrent_gpus']==3 and key not in s['jobs']
    before=sum(float(j.get('charged_gpu_hours') or 0) for j in s['jobs'].values())
    backup=P/'ledger_before_reconciliation_PRIVATE.json'
    with backup.open('x') as stream:json.dump(s,stream,indent=2)
    migrated=[]
    for name,j in s['jobs'].items():
        if 'state' not in j:
            assert isinstance(j.get('charged_gpu_hours'),(float,int)) and j['charged_gpu_hours']>=0
            j['state']='FINISHED';j['state_reconciliation']='Existing measured manual charge retained; added canonical FINISHED status for strict budget summary.'
            migrated.append(name)
    s['jobs'][key]={'state':'FINISHED','gpus':3,'start_unix':start,'end_unix':end,
        'charged_gpu_hours':upper,'reserved_gpu_hours':upper,'reservation_overrun':False,
        'accounting_method':'CONSERVATIVE_UPPER_BOUND: all three recorded cards held for entire first-to-last foreground GPU command window, including gaps; not claimed measured actual compute.',
        'historical_execution_metadata_sha256':hashlib.sha256(history.read_bytes()).hexdigest(),
        'reason':'Missing old TEST/sanity accounting; CPU-only earlier checks excluded by explicit disabled CUDA binding. No historical model command rerun.'}
    after=sum(float(j.get('charged_gpu_hours') or 0) for j in s['jobs'].values())
    active=[j for j in s['jobs'].values() if j.get('state') in ['ACTIVE','RUNNING','RESERVED']]
    assert not active and after+2<=72
    temp=L.with_name(L.name+'.source_tail_reconcile.tmp');temp.write_text(json.dumps(s,indent=2)+'\n');os.replace(temp,L)
result={'status':'PASS_CONSERVATIVE_RECONCILIATION','ledger_path':str(L),'cap_gpu_hours':72,
    'previous_charged_gpu_hours':before,'historical_conservative_charge_gpu_hours':upper,
    'charged_gpu_hours_after':after,'available_lower_bound_gpu_hours':72-after,
    'new_cohort_cap_gpu_hours':2,'maximum_new_cohort_gpus':1,'ledger_format_only_status_migrations':migrated,
    'old_charges_unchanged':True,'historical_actual_elapsed_not_claimed':True,'accounting_window_start_utc':datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),
    'accounting_window_end_utc':datetime.datetime.fromtimestamp(end,datetime.timezone.utc).isoformat(),
    'maximum_historical_cards':3,'historical_gpu_command_events':events,'raw_command_records_private':True}
(P/'BUDGET_RECONCILIATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['historical_gpu_command_events','ledger_format_only_status_migrations']},indent=2))
