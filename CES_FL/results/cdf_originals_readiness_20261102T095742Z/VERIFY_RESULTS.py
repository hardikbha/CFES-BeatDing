import json,hashlib,datetime,subprocess
from pathlib import Path
import torch
root=Path(__file__).resolve().parent
p=Path('CES_FL/proposals/cdf_response_20261102_r1');manifest=json.loads((p/'SOURCE_MANIFEST.json').read_text())
def sha(file):
 h=hashlib.sha256()
 with Path(file).open('rb') as stream:
  for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
 return h.hexdigest()
for name,expected in manifest['protected_source_and_config_sha256'].items():assert sha(name)==expected,name
records={};runs=Path('/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z')
for arm in ['bce','fair']:
 d=runs/arm;receipt=json.loads((d/'RUN_VERIFIED.json').read_text());history=[json.loads(s) for s in (d/'history.jsonl').read_text().splitlines() if s.strip()]
 assert (d/'DONE').is_file() and receipt['status']=='PASS' and receipt['validation_replayed'] and not receipt['test_read']
 assert [v['epoch'] for v in history]==list(range(1,51))
 selected=max(history,key=lambda v:v['validation']['overall']['auc'])
 actual=sha(d/'best.pt');assert actual==receipt['checkpoint_sha256']
 ck=torch.load(d/'best.pt',map_location='cpu',weights_only=True)
 assert ck['epoch']==selected['epoch'] and ck['validation']==selected['validation']
 assert ck['config']['pretrain']['seed']==17 and ck['arm']==arm
 assert ck['hashes']==receipt['hashes'] and not ck['synthetic_fixture'] and not ck['quartet_usage']
 assert all(torch.isfinite(v).all() for v in ck['model'].values())
 records[arm]={'root':str(d),'status':'COMPLETED_VERIFIED','epochs':len(history),'selected_epoch':selected['epoch'],'validation':selected['validation'],
 'calibration':ck['calibration'],'fresh_checkpoint_sha256':actual,'native_replay_receipt':receipt,'checkpoint_metadata_and_history_agree':True,'fresh_full_validation_inference':False,
 'config_sha256':sha(d/'config.json'),'source_commit':'58773bbef9d9fe79dd313610f694ddeea12c6037','pair_draws':0}
 del ck
configs={}
for seed in [7,17,27]:
 f=p/f'configs/xception_s{seed}.json';c=json.loads(f.read_text());configs[str(seed)]={'config_sha256':sha(f),'pretrain_run':c['pretrain_run'],'pretrain_sha256':c['pretrain_sha256'],'bank':c['bank']}
ff=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1/training');f=json.loads((ff/'RUN_VERIFIED.json').read_text())
assert (ff/'DONE').is_file() and f['status']=='PASS' and f['completed_epochs']==10 and f['native_independent_validation_replay'] and not f['test_evaluated']
out={'observed_management_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'latest_fetched_main':subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip(),'protected_source_config_files_verified':len(manifest['protected_source_and_config_sha256']),
 'records':records,'proposed_Xception_bindings':configs,'new_control_training_performed':False,'TEST_data_read':False,
 'new_per_arm_GPU_checks':'BLOCKED_GPU_RESERVATIONS;not executed;existing source-matched CUDA smoke remains historical evidence',
 'ff2_existing_run':{'root':str(ff),'DONE':True,'status':f['status'],'epochs':f['completed_epochs'],'selected_epoch':f['selected_epoch'],'validation_auc':f['validation_auc'],
 'native_validation_replay':True,'test_evaluated':False,'further_work':'HOLD','checkpoint_hash_from_native_receipt':f['checkpoint_sha256'],'checkpoint_rehashed_by_this_FF2_check':False}}
(root/'VERIFIED_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':'PASS','out':str(root),'source_files_verified':out['protected_source_config_files_verified'],'epochs':{a:r['epochs'] for a,r in records.items()},'selected_epochs':{a:r['selected_epoch'] for a,r in records.items()},'FF2':out['ff2_existing_run']},indent=2))
