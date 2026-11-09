"""One authorized pool-v2 lease, existing shared ledger, strict two-hour cohort."""
import argparse,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path

def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root',required=True);a=p.parse_args()
src=Path(a.source);root=Path(a.root);root.joinpath('logs').mkdir(exist_ok=True)
pool=module('/home/sachin.chaudhary/ces_fl_runs/ff2_baseline_suite_s17_20261103T110324Z/launch_pool_method_v2.py','authorized_pool')
budget=module(src/'CES_FL/proposals/ff2_baseline_suite_s17_20261103T110324Z/budget.py','shared_budget')
ledger=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_baseline_suite_s17_20261103T110324Z/additional72_gpu_hours.json')
git='/home/soft/anaconda3/pkgs/git-2.45.2-pl5340h9abc3c3_1/bin/git'
assert subprocess.check_output([git,'-C',str(src),'rev-parse','HEAD'],text=True).strip()=='9bb8e8332b8a3aa78d83f74ef21bca0b7dcd6754'
assert subprocess.run([git,'-C',str(src),'symbolic-ref','-q','HEAD'],capture_output=True).returncode!=0
assert not subprocess.check_output([git,'-C',str(src),'status','--porcelain'],text=True).strip()
assert json.loads((root/'CPU_INPUT_BINDINGS.json').read_text())['status']=='PASS'
assert json.loads((root/'BUDGET_RECONCILIATION.json').read_text())['status']=='PASS_CONSERVATIVE_RECONCILIATION'
session='source_tail_20261109_r1_cohort1';entry=None;child=None;claimed=None;status='FAILED';rc=None
def write(p,d):p=Path(p);p.write_text(json.dumps(d,indent=2)+'\n')
def interrupted(*_):raise KeyboardInterrupt('Own source-tail cohort interrupted')
for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
try:
    with pool.mutex():
        # The established user-authorized shared route reserves capacity, not
        # exclusive ownership or a guarantee of idle utilization. candidates()
        # already honors protected tasks and every live pool reservation.
        available=[c for c in pool.candidates() if c['available_mib_after_other_leases']>=30720]
        if not available:raise RuntimeError('No authorized shared capacity; no job displaced')
        c=available[0];directory=pool.LOCKS/f"gpu{c['index']}"
        if not directory.exists():directory.mkdir();pool.write(directory/'pool_owner.json',{'shared':True,'v':2,'pid':0,'wid':'pool_v2_shared'})
        entry=directory/f'task_ff2_native_{session}.json'
        if entry.exists():raise RuntimeError('Existing cohort lease: do not duplicate')
        claimed=time.time();clock=time.monotonic()
        pool.write(entry,{'name':'ff2_native_'+session,'wid':'ff2_native_'+session,'pid':os.getpid(),'pgid':None,'cls':'reader','need':pool.NEED,'time':claimed})
        ownership={'lease_path':str(entry),'lease':pool.read(entry),'pool_owner':pool.read(directory/'pool_owner.json'),'card':c,
                   'host':os.uname().nodename,'route':'Previously explicitly user-authorized direct-SSH shared-pool-v2','pbs_allocation_claimed':False}
        write(root/'OWNERSHIP_PRIVATE.json',ownership)
    allocation={'allocation_id':f"pool-v2:{os.uname().nodename}:{c['uuid']}:{session}",'gpus':1,'cuda_visible_devices':c['uuid'],
                'verified_by':'operator, existing pool-v2 candidate/lease guards and prior user SSH/shared authorization',
                'evidence':str(root/'OWNERSHIP_PRIVATE.json'),'pbs_allocation_claimed':False,'exclusive_physical_ownership_claimed':False}
    write(root/'ALLOCATION.json',allocation)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=c['uuid'],PATH=str(Path(git).parent)+':'+env.get('PATH',''),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1')
    cmd=[sys.executable,'-u',str(src/'CES_FL/proposals/ff2_expanded_baselines_20261104_r1/with_gpu_budget.py'),
         '--ledger',str(ledger),'--allocation',str(root/'ALLOCATION.json'),'--session',session,'--hours','1.98','--out',str(root/'budget_session'),
         '--',sys.executable,'-u',str(root/'runtime/cohort.py'),'--source',str(src),'--root',str(root)]
    write(root/'COMMAND.json',{'argv':cmd,'source_commit':'9bb8e8332b8a3aa78d83f74ef21bca0b7dcd6754','cohort_cap_gpu_hours':2,'wrapper_reservation_gpu_hours':1.98,'all_cuda_stages_inside_guard':True})
    with (root/'logs/budget_controller.log').open('x') as log:
        child=subprocess.Popen(cmd,env=env,cwd=src,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with pool.mutex():x=pool.read(entry);x['pgid']=child.pid;pool.write(entry,x)
        write(root/'PROCESS.json',{'controller_pid':os.getpid(),'wrapper_pid':child.pid,'host':os.uname().nodename,'gpu_uuid':c['uuid'],'gpu_index':c['index'],'start_node_unix':claimed})
        rc=child.wait(timeout=max(1,2*3600-(time.monotonic()-clock)-10))
    status='COMPLETED' if rc==0 else 'FAILED'
finally:
    if child is not None and child.poll() is None:
        # Only this explicitly created process group is signalled.
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    if entry is not None:
        with pool.mutex():
            if entry.exists() and pool.read(entry).get('pid')==os.getpid():entry.unlink()
    if claimed is not None:
        end=time.time();held=(end-claimed)/3600
        with budget.locked(ledger) as state:
            j=state['jobs'].get(session)
            if j and j.get('state')=='ACTIVE':
                j.update(state='FINISHED',end_unix=end,charged_gpu_hours=(end-j['start_unix'])/3600,reservation_overrun=False,operator_finalized_after_wrapper_failure=True)
            inner=float((j or {}).get('charged_gpu_hours') or 0);bookend=max(0.,held-inner)
            state['jobs'][session+'_bookends']={'state':'FINISHED','gpus':1,'start_unix':claimed,'end_unix':end,'charged_gpu_hours':bookend,
                'scope':'Actual pool-lease startup/cleanup/held idle not included in inner command wall-time; not extra GPU usage grant.'}
        write(root/'COHORT_ALLOCATION_END.json',{'status':status,'exit_code':rc,'start_node_unix':claimed,'end_node_unix':end,'held_gpu_hours':held,
              'inner_charged_gpu_hours':inner,'bookend_gpu_hours':bookend,'cap_gpu_hours':2,'cap_exceeded':held>2,
              'gpu_uuid':c['uuid'],'host':os.uname().nodename,'max_gpus':1,'no_automatic_extension':True})
if rc:raise SystemExit(rc)
