#!/usr/bin/env python3
"""Run one frozen ResNet L_A/L_H Table-3 evaluation pass on a leased GPU."""
import contextlib
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time

PYTHON='/home/sachin.chaudhary/.conda/envs/aiface/bin/python'
SOURCE=Path(__file__).resolve().parents[2]
LOCKS=Path('/tmp/scf_gpu_locks')
BUDGET_FILE=Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/allocated_stage.py')
RESERVED_HOURS=1.0
DATASETS=('ffpp','dfdc','celebdf','dfd')


def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    p=Path(path)
    if p.exists():raise ValueError(f'Preserve existing output: {p}')
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,p)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


@contextlib.contextmanager
def pool_mutex():
    LOCKS.mkdir(exist_ok=True);p=LOCKS/'.pool_mutex';deadline=time.monotonic()+30
    while True:
        try:p.mkdir();break
        except FileExistsError:
            if time.monotonic()>deadline:raise RuntimeError('Direct GPU lease mutex busy')
            time.sleep(.2)
    try:yield
    finally:p.rmdir()


def card_by_uuid(uuid):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'],text=True)
    cards=[{'index':int(r[0]),'uuid':r[1].strip(),'free_mib':int(r[2]),'used_mib':int(r[3]),'utilization':int(r[4])}
           for r in csv.reader(output.splitlines())]
    found=[c for c in cards if c['uuid']==uuid]
    if len(found)!=1:raise RuntimeError('GPU UUID absent or ambiguous')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                  '--format=csv,noheader,nounits'],text=True)
    active=[r for r in csv.reader(apps.splitlines()) if r and r[0].strip()==uuid]
    if found[0]['free_mib']<24000 or active:raise RuntimeError(f'GPU lacks free evaluation headroom: {found[0]} apps={active}')
    return found[0]


def verify_binding(torch,uuid):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected one visible H100 device')
    torch.empty(1,device='cuda')
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                 '--format=csv,noheader,nounits'],text=True)
    rows=[r for r in csv.reader(out.splitlines()) if len(r)>=2 and r[1].strip()==str(os.getpid())]
    if not any(r[0].strip()==uuid for r in rows):raise RuntimeError(f'CUDA UUID binding mismatch: {rows}')
    return rows


def reserve(budget,sid,card,commit,log):
    with budget.ledger() as d:
        if any(x['id']==sid for x in d['gpu_sessions']):raise RuntimeError('Duplicate evaluation reservation')
        committed=sum(x['reserved_gpu_hours'] if x['state'] in ('QUEUED','RUNNING')
                      else x.get('elapsed_seconds',0)*x['ngpus']/3600 for x in d['gpu_sessions'])
        if committed+RESERVED_HOURS>d['initial_total_gpu_hours']:raise RuntimeError('Evaluation exceeds shared budget')
        d['gpu_sessions'].append({'id':sid,'ngpus':1,'reserved_gpu_hours':RESERVED_HOURS,'elapsed_seconds':0.,
            'state':'RUNNING','route':'user_authorized_direct_ssh_pool_v2','pbs_allocation_claimed':False,
            'host':os.uname().nodename,'controller_pid':os.getpid(),'card':card,
            'scientific_commit':commit,'purpose':'single locked ResNet Table-3 inference pass; no calibration',
            'log':str(log)})


def main():
    parser=__import__('argparse').ArgumentParser()
    parser.add_argument('--arm',choices=('L_A','L_H'),required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--attempt-id',required=True)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--out-root',required=True)
    args=parser.parse_args()
    if os.uname().nodename.split('.')[0] not in ('gpu01','gpu02'):raise RuntimeError('Use SSH gpu01/gpu02')
    if not args.gpu_uuid.startswith('GPU-') or not args.attempt_id.replace('_','').replace('-','').isalnum():
        raise ValueError('Invalid UUID or attempt identifier')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    outroot=Path(args.out_root).resolve();plan_path=Path(args.plan).resolve()
    if outroot.exists() and any((outroot/args.arm/d).exists() for d in DATASETS):
        raise ValueError('One or more dataset outputs already exist; preserve and do not repeat TEST')
    git='/home/soft/anaconda3/pkgs/git-2.45.2-pl5340h9abc3c3_1/bin/git'
    commit=subprocess.check_output([git,'-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if subprocess.run([git,'-C',str(SOURCE),'symbolic-ref','-q','HEAD'],stdout=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('Evaluation source checkout must be detached')
    if subprocess.check_output([git,'-C',str(SOURCE),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Evaluation checkout is dirty')
    locked=load('cesfl_table3_locked',SOURCE/'CES_FL/scripts/lh_resnet50_locked_table3.py')
    _,_,plan=locked.load_plan(plan_path)
    if args.arm not in plan['arms'] or tuple(plan['datasets'])!=DATASETS:
        raise ValueError('Wrong arm/dataset inventory in locked plan')
    for dataset in DATASETS:
        if (outroot/args.arm/dataset).exists():raise ValueError(f'Existing TEST output: {outroot/args.arm/dataset}')
    gpu=card_by_uuid(args.gpu_uuid)
    budget=load('cesfl_direct_budget',BUDGET_FILE)
    sid=f'lh_resnet50_table3_eval_{args.arm}_s17_{args.attempt_id}'
    lease_dir=LOCKS/f"gpu{gpu['index']}";lease=lease_dir/f'task_{sid}.json'
    with pool_mutex():
        gpu=card_by_uuid(args.gpu_uuid)
        if lease.exists():raise RuntimeError('Duplicate direct evaluation lease')
        if lease_dir.exists():
            owner=lease_dir/'pool_owner.json'
            if not owner.exists() or not read_json(owner).get('shared'):raise RuntimeError('GPU is not in shared pool-v2')
            active=[read_json(p) for p in lease_dir.glob('task_*.json')
                    if isinstance(read_json(p).get('pid'),int) and Path(f"/proc/{read_json(p).get('pid')}").exists()]
            if active:raise RuntimeError(f'GPU already leased: {active}')
        else:
            lease_dir.mkdir();write_json(lease_dir/'pool_owner.json',{'shared':True,'v':2,'pid':0,'wid':'pool_v2_shared'})
        write_json(lease,{'name':sid,'wid':sid,'pid':os.getpid(),'pgid':os.getpgrp(),'cls':'writer',
                          'need':24000,'time':time.time(),'device_uuid':gpu['uuid'],'commit':commit})
    log=outroot/f'{args.arm}.direct_eval.log';outroot.mkdir(parents=True,exist_ok=True)
    try:reserve(budget,sid,gpu,commit,log)
    except Exception:
        with pool_mutex():
            if lease.exists() and read_json(lease).get('pid')==os.getpid():lease.unlink()
        raise
    started=time.monotonic();stop=threading.Event();state={'status':'FAILED','completed_datasets':[]}
    def heartbeat():
        while not stop.wait(20):
            with budget.ledger() as d:
                row=next(x for x in d['gpu_sessions'] if x['id']==sid);row['elapsed_seconds']=time.monotonic()-started
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        import torch
        binding=verify_binding(torch,gpu['uuid'])
        write_json(outroot/f'{args.arm}_CUDA_BINDING.json',{'status':'PASS','pid':os.getpid(),
            'expected_device_uuid':gpu['uuid'],'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
            'device_name':torch.cuda.get_device_name(0),'nvidia_smi_process_rows':binding,
            'executor_commit':commit,'plan_sha256':sha(plan_path),'test_read':False})
        torch.set_num_threads(4)
        state={'status':'RUNNING','arm':args.arm,'plan_sha256':sha(plan_path),'completed_datasets':[],'test_read':True}
        for dataset in DATASETS:
            dataset_out=outroot/args.arm/dataset
            print(json.dumps({'status':'INFERENCE_START','arm':args.arm,'dataset':dataset}),flush=True)
            result=locked.inference(plan_path,args.arm,dataset,dataset_out,torch.device('cuda'))
            state['completed_datasets'].append({'dataset':dataset,'checkpoint_sha256':result['checkpoint_sha256'],
                'manifest_sha256':result['manifest_sha256'],'metrics_sha256':sha(dataset_out/'metrics.json'),
                'scores_sha256':sha(dataset_out/'scores_PRIVATE.json')})
            print(json.dumps({'status':'INFERENCE_PASS','arm':args.arm,'dataset':dataset,
                              'auc':result['calibrated']['overall']['auc']}),flush=True)
        state['status']='COMPLETED';state['test_threshold_search']=False;state['target_domain_calibration']=False
    except Exception:
        if state.get('status')=='RUNNING':state['status']='FAILED_PARTIAL'
        raise
    finally:
        stop.set();thread.join(timeout=20);elapsed=time.monotonic()-started
        state.update(elapsed_seconds=elapsed,gpu_hours=elapsed/3600,gpu=gpu,host=os.uname().nodename,executor_commit=commit)
        write_json(outroot/f'{args.arm}_ALLOCATION_END.json',state)
        with budget.ledger() as d:
            row=next(x for x in d['gpu_sessions'] if x['id']==sid);row.update(state=state['status'],elapsed_seconds=elapsed)
        with pool_mutex():
            if lease.exists() and read_json(lease).get('pid')==os.getpid():lease.unlink()


if __name__=='__main__':main()
