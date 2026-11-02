import csv,json,os,pathlib,socket,subprocess,sys,time
host=socket.gethostname().split('.')[0]
root=pathlib.Path('/home/sachin.chaudhary/cesel_ma_audit_logs')
source=pathlib.Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-tail-response-20261102-r1')
out=pathlib.Path('/home/sachin.chaudhary/cesel_ma_audit_logs/cdf_followup_check_20261102T095742Z/real_checks')
assert not out.exists()
capture=lambda argv:subprocess.check_output(argv,text=True)
occupied=set(capture(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader']).splitlines())
devices=list(csv.reader(capture(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits']).splitlines()))
registry=lease=None
try:
 for row in sorted(devices,key=lambda x:-int(x[2])):
  index,uuid,free,util=[x.strip() for x in row]
  if uuid in occupied or int(free)<65000 or int(util)>5:continue
  registry=pathlib.Path('/tmp/scf_gpu_locks')/('gpu'+index);registry.parent.mkdir(exist_ok=True)
  try:registry.mkdir()
  except FileExistsError:registry=None;continue
  (registry/'cesfl_owner.json').write_text(json.dumps({'role':'CES_ORIGINALS_REAL_BACKWARD_ONLY','pid':os.getpid()}))
  lease=pathlib.Path('/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA/work/gpulocks')/(host+'_'+index)
  try:lease.mkdir()
  except FileExistsError:
   lease=None;(registry/'cesfl_owner.json').unlink();registry.rmdir();registry=None;continue
  (lease/'by').write_text('CDF synthetic software smoke '+str(os.getpid()))
  now=set(capture(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader']).splitlines())
  assert uuid not in now,'Device became occupied; stop instead of sharing'
  break
 assert lease is not None,'No idle cooperatively available GPU; no other job modified'
 binding={'host':host,'physical_index':int(index),'gpu_uuid':uuid,'free_mib':int(free),'memory_fraction':.45,'pbs_allocation':None,'execution_route':'user-authorized direct SSH software smoke, cooperative idle GPU','no_existing_process_stopped':True}
 out.mkdir(exist_ok=False)
 (out/'BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
 records=[]
 for arm in ['bce','dro','mean_fair']:
  argv=['ces_fl_cdf_study.py','dry-run','--config',str(source/'CES_FL/proposals/cdf_response_20261102_r1/configs/xception_s17.json'),'--arm',arm,'--device','cuda','--out',str(out/arm)]
  code="import runpy,sys,torch;sys.path.insert(0,"+repr(str(source/'CES_FL/src'))+");assert torch.cuda.device_count()==1;torch.cuda.set_per_process_memory_fraction(.45,0);sys.argv="+repr(argv)+";runpy.run_path("+repr(str(source/'CES_FL/src/ces_fl_cdf_study.py'))+",run_name='__main__')"
  command=['/home/sachin.chaudhary/.conda/envs/qwen38/bin/python','-u','-c',code]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
  print('ARM_BEGIN',arm,flush=True);start=time.monotonic()
  with (out/(arm+'.log')).open('x') as log:
   result=subprocess.run(command,env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT)
  records.append({'arm':arm,'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start,'command':command})
  (out/'COMMANDS.json').write_text(json.dumps({'binding':binding,'checks':records},indent=2)+'\n')
  print('ARM_END',json.dumps(records[-1]),flush=True)
  if result.returncode:raise SystemExit(result.returncode)
  receipt=json.loads((out/arm/'DRY_RUN.json').read_text())
  assert receipt['status']=='PASS' and receipt['optimizer_steps']==0 and receipt['synthetic_fixture'] is False
 print('ALL_THREE_REAL_CHECKS_PASS',flush=True)

finally:
 if lease is not None and (lease/'by').exists():(lease/'by').unlink();lease.rmdir()
 if registry is not None and (registry/'cesfl_owner.json').exists():(registry/'cesfl_owner.json').unlink();registry.rmdir()
