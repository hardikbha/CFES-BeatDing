import csv,json,os,pathlib,socket,subprocess,sys,time
host=socket.gethostname().split('.')[0]
root=pathlib.Path('/home/sachin.chaudhary/cesel_ma_audit_logs')
source=pathlib.Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-tail-response-20261102-r1')
out=root/'cdf_response_20261102_r1_cuda_v6'
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
  (registry/'cesfl_owner.json').write_text(json.dumps({'role':'synthetic_CDF_smoke','pid':os.getpid()}))
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
 (root/'cdf_response_20261102_r1_cuda_binding_v6.json').write_text(json.dumps(binding,indent=2)+'\n')
 command=['/home/sachin.chaudhary/.conda/envs/qwen38/bin/python','-u','-c',
  "import runpy,sys,torch;assert torch.cuda.device_count()==1;torch.cuda.set_per_process_memory_fraction(.45,0);sys.argv=['smoke_cdf_proposal.py','--device','cuda','--out',"+repr(str(out))+"];runpy.run_path("+repr(str(source/'CES_FL/tests/smoke_cdf_proposal.py'))+",run_name='__main__')"]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 print('BOUND',json.dumps(binding),flush=True)
 start=time.monotonic();p=subprocess.run(command,env=env,cwd=source)
 (root/'cdf_response_20261102_r1_cuda_exit_v6.json').write_text(json.dumps({'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-start,'command':command,**binding},indent=2)+'\n')
 raise SystemExit(p.returncode)
finally:
 if lease is not None and (lease/'by').exists():(lease/'by').unlink();lease.rmdir()
 if registry is not None and (registry/'cesfl_owner.json').exists():(registry/'cesfl_owner.json').unlink();registry.rmdir()
