#!/usr/bin/env python3
"""Run the locked ResNet Table-3 inference under a PBS ngpus=1 allocation."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

SOURCE=Path(__file__).resolve().parents[2]
BUDGET_FILE=Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/allocated_stage.py')
DATASETS=('ffpp','dfdc','celebdf','dfd')
ARMS=('L_A','L_H')
RESERVED_HOURS=2.0


def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value,replace=False):
    p=Path(path)
    if p.exists() and not replace:raise ValueError(f'Preserve existing output: {p}')
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,p)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def pbs_allocation(jobid):
    qstat='/opt/pbs/bin/qstat'
    raw=subprocess.check_output([qstat,'-f',jobid],text=True)
    select=re.search(r'Resource_List\.select\s*=\s*([^\n]+)',raw)
    vnode=re.search(r'exec_vnode\s*=\s*([^\n]+)',raw)
    host=re.search(r'exec_host\s*=\s*([^\n]+)',raw)
    if not select or not vnode or not host or 'ngpus=1' not in select.group(1) or 'ngpus=1' not in vnode.group(1):
        raise RuntimeError('PBS qstat does not verify an allocated ngpus=1 vnode')
    return {'qstat_select':select.group(1).strip(),'exec_vnode':vnode.group(1).strip(),
            'exec_host':host.group(1).strip(),'queue_resource_verified':True}


def verify_cuda(torch,expected_host):
    cvd=os.environ.get('CUDA_VISIBLE_DEVICES','').strip()
    if not cvd:raise RuntimeError('PBS did not provide CUDA_VISIBLE_DEVICES; refusing to guess')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('PBS CUDA binding did not expose exactly one device')
    name=torch.cuda.get_device_name(0)
    if 'H100' not in name:raise RuntimeError(f'Expected H100, received {name}')
    torch.empty(1,device='cuda')
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                    '--format=csv,noheader,nounits'],text=True)
    own=[r for r in csv.reader(output.splitlines()) if len(r)>=2 and r[1].strip()==str(os.getpid())]
    uuids={r[0].strip() for r in own}
    if len(uuids)!=1:raise RuntimeError(f'PBS PID does not map to exactly one GPU UUID: {own}')
    return {'status':'PASS','pbs_jobid':os.environ['PBS_JOBID'],'host':os.uname().nodename,
            'expected_host':expected_host,'cuda_visible_devices':cvd,'device_name':name,
            'process_gpu_rows':own,'device_uuid':next(iter(uuids)),'allocation_binding_verified':True}


def reserve(budget,sid,jobid,log):
    with budget.ledger() as d:
        if any(x['id']==sid for x in d['gpu_sessions']):raise RuntimeError('Duplicate PBS evaluation budget entry')
        committed=sum(x['reserved_gpu_hours'] if x['state'] in ('QUEUED','RUNNING')
                      else x.get('elapsed_seconds',0)*x['ngpus']/3600 for x in d['gpu_sessions'])
        if committed+RESERVED_HOURS>d['initial_total_gpu_hours']:
            raise RuntimeError('PBS evaluation exceeds the authorized shared budget')
        d['gpu_sessions'].append({'id':sid,'ngpus':1,'reserved_gpu_hours':RESERVED_HOURS,'elapsed_seconds':0.,
            'state':'RUNNING','route':'PBS_ngpus1_shared','pbs_allocation_claimed':True,
            'pbs_job_id':jobid,'host':os.uname().nodename,'controller_pid':os.getpid(),
            'log':str(log),'purpose':'one locked ResNet Table-3 pass, both frozen arms; no calibration/search'})


def main():
    parser=__import__('argparse').ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('--out-root',required=True)
    parser.add_argument('--expected-host',choices=('gpu01','gpu02'),required=True)
    args=parser.parse_args()
    jobid=os.environ.get('PBS_JOBID')
    if not jobid:raise RuntimeError('PBS_JOBID missing; this script must run inside PBS')
    if os.uname().nodename.split('.')[0]!=args.expected_host:
        raise RuntimeError(f'PBS placement mismatch: expected {args.expected_host}, got {os.uname().nodename}')
    alloc=pbs_allocation(jobid)
    if args.expected_host not in alloc['exec_host'] or args.expected_host not in alloc['exec_vnode']:
        raise RuntimeError('PBS exec_host/vnode does not match requested GPU host')
    root=Path(args.out_root).resolve();plan_path=Path(args.plan).resolve()
    if not root.is_dir():raise RuntimeError('Locked Table-3 output root is absent')
    if any((root/arm/dataset).exists() for arm in ARMS for dataset in DATASETS):
        raise RuntimeError('Some TEST outputs already exist; preserve and do not rerun')
    git='/home/soft/anaconda3/pkgs/git-2.45.2-pl5340h9abc3c3_1/bin/git'
    commit=subprocess.check_output([git,'-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if subprocess.run([git,'-C',str(SOURCE),'symbolic-ref','-q','HEAD'],stdout=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('PBS evaluator source checkout must be detached')
    if subprocess.check_output([git,'-C',str(SOURCE),'status','--porcelain'],text=True).strip():
        raise RuntimeError('PBS evaluator checkout is dirty')
    # CVD is inherited from PBS before importing the evaluator/torch modules.
    locked=load('cesfl_table3_locked',SOURCE/'CES_FL/scripts/lh_resnet50_locked_table3.py')
    _,_,plan=locked.load_plan(plan_path)
    if set(plan['arms'])!=set(ARMS) or tuple(plan['datasets'])!=DATASETS:
        raise ValueError('Locked evaluation plan does not bind the two arms/four datasets')
    for arm in ARMS:
        locked.verify_selected(locked.load_core(),plan['arms'][arm]['run'],arm)
    # No TEST manifest is opened before this point; this is the immutable-plan check.
    budget=load('cesfl_pbs_budget',BUDGET_FILE);sid=f'lh_resnet50_table3_eval_{jobid}'
    log=root/'PBS_TABLE3_EVAL.log';reserve(budget,sid,jobid,log)
    started=time.monotonic();stop=threading.Event();state={'status':'FAILED','completed':[],'test_read':False}
    def heartbeat():
        while not stop.wait(20):
            with budget.ledger() as d:
                x=next(z for z in d['gpu_sessions'] if z['id']==sid);x['elapsed_seconds']=time.monotonic()-started
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        import torch
        binding=verify_cuda(torch,args.expected_host)
        write_json(root/'PBS_CUDA_BINDING.json',{**binding,**alloc,'executor_commit':commit,
            'plan_sha256':sha(plan_path),'test_read':False})
        torch.set_num_threads(4)
        state={'status':'RUNNING','completed':[],'test_read':False,'pbs_jobid':jobid,'plan_sha256':sha(plan_path)}
        for arm in ARMS:
            for dataset in DATASETS:
                out=root/arm/dataset
                if out.exists():raise RuntimeError(f'Preserve existing inference output {out}')
                print(json.dumps({'status':'INFERENCE_START','arm':arm,'dataset':dataset}),flush=True)
                state['test_read']=True
                result=locked.inference(plan_path,arm,dataset,out,torch.device('cuda'))
                item={'arm':arm,'dataset':dataset,'checkpoint_sha256':result['checkpoint_sha256'],
                      'manifest_sha256':result['manifest_sha256'],'metrics_sha256':sha(out/'metrics.json'),
                      'scores_sha256':sha(out/'scores_PRIVATE.json')}
                state['completed'].append(item)
                write_json(root/'PBS_PROGRESS.json',state,replace=True)
                print(json.dumps({'status':'INFERENCE_PASS','arm':arm,'dataset':dataset,
                    'auc':result['calibrated']['overall']['auc']}),flush=True)
        state.update(status='COMPLETED',test_threshold_search=False,target_domain_calibration=False)
    except Exception:
        if state.get('status')=='RUNNING':state['status']='FAILED_PARTIAL'
        raise
    finally:
        stop.set();thread.join(timeout=20);elapsed=time.monotonic()-started
        state.update(elapsed_seconds=elapsed,gpu_hours=elapsed/3600,executor_commit=commit,allocation=alloc)
        write_json(root/'PBS_ALLOCATION_END.json',state)
        with budget.ledger() as d:
            x=next(z for z in d['gpu_sessions'] if z['id']==sid);x.update(state=state['status'],elapsed_seconds=elapsed)


if __name__=='__main__':main()
