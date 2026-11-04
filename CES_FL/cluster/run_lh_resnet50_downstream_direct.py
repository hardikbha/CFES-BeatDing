#!/usr/bin/env python3
"""Source-bound direct-SSH runner for one matched ResNet-50 L_A or L_H arm."""
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

PYTHON='/home/sachin.chaudhary/.conda/envs/aiface/bin/python'
SOURCE=Path(__file__).resolve().parents[2]
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/lh_resnet50_table3_s17_20261104')
LOCKS=Path('/tmp/scf_gpu_locks')
BUDGET_MODULE=Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/allocated_stage.py')
HEAD_SHA='72d368a439658c601a5fe4d717237b1561190e85c13c70f6db3a441bc5cdbe78'
P50_DIR=ROOT/'P50'
RESERVED_HOURS=2.0


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    p=Path(path);tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


@contextlib.contextmanager
def pool_mutex():
    LOCKS.mkdir(exist_ok=True);p=LOCKS/'.pool_mutex';deadline=time.monotonic()+30
    while True:
        try:p.mkdir();break
        except FileExistsError:
            if time.monotonic()>deadline:raise RuntimeError('GPU lease mutex busy')
            time.sleep(.2)
    try:yield
    finally:p.rmdir()


def card_by_uuid(uuid):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    cards=[]
    for row in csv.reader(output.splitlines()):
        cards.append({'index':int(row[0]),'uuid':row[1].strip(),'free_mib':int(row[2]),
                      'used_mib':int(row[3]),'utilization':int(row[4])})
    found=[c for c in cards if c['uuid']==uuid]
    if len(found)!=1:raise RuntimeError('GPU UUID absent/ambiguous')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    active=[r for r in csv.reader(apps.splitlines()) if r and r[0].strip()==uuid]
    if found[0]['free_mib']<24000 or active:raise RuntimeError(f'GPU lacks free direct-lease headroom: {found[0]} apps={active}')
    return found[0]


def verify_cuda_uuid(torch, expected_uuid):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Direct CUDA binding did not expose exactly one GPU')
    if 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Direct CUDA binding exposed an unexpected GPU model')
    torch.empty(1,device='cuda')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory',
                                  '--format=csv,noheader,nounits'],text=True)
    rows=[r for r in csv.reader(apps.splitlines()) if len(r)>=2 and r[1].strip()==str(os.getpid())]
    if not any(r[0].strip()==expected_uuid for r in rows):
        raise RuntimeError(f'CUDA process is not mapped to leased UUID {expected_uuid}: {rows}')


def reserve(budget,sid,card,head,commit):
    with budget.ledger() as d:
        if any(s['id']==sid for s in d['gpu_sessions']):raise RuntimeError('Duplicate arm reservation')
        committed=sum(s['reserved_gpu_hours'] if s['state'] in ('QUEUED','RUNNING')
                      else s.get('elapsed_seconds',0)*s['ngpus']/3600 for s in d['gpu_sessions'])
        if committed+RESERVED_HOURS>d['initial_total_gpu_hours']:raise RuntimeError('Arm exceeds extended study budget')
        d['gpu_sessions'].append({'id':sid,'ngpus':1,'reserved_gpu_hours':RESERVED_HOURS,
            'elapsed_seconds':0.,'state':'RUNNING','route':'user_authorized_direct_ssh_pool_v2',
            'pbs_allocation_claimed':False,'host':os.uname().nodename,'controller_pid':os.getpid(),
            'card':card,'scientific_commit':commit,'scientific_source_sha256':head.sha(head.__file__),
            'log':str(ROOT/f'{sid}.log'),'purpose':f'ResNet-50 Table3 {sid} arm; no automatic TEST evaluation'})


def main():
    parser=__import__('argparse').ArgumentParser()
    parser.add_argument('--arm',choices=('L_A','L_H'),required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--budget-hours',type=float,default=RESERVED_HOURS)
    args=parser.parse_args()
    if args.budget_hours!=RESERVED_HOURS:raise ValueError('Fixed three-hour per-arm safety reservation expected')
    if os.uname().nodename.split('.')[0] not in ('gpu01','gpu02'):raise RuntimeError('Direct SSH must land on gpu01 or gpu02')
    if not (P50_DIR/'DONE').is_file() or not (P50_DIR/'RUN_VERIFIED.json').is_file():raise RuntimeError('Verified P50 dependency absent')
    p50_receipt=read_json(P50_DIR/'RUN_VERIFIED.json')
    if p50_receipt.get('status')!='PASS' or p50_receipt.get('test_read') is not False:
        raise RuntimeError('P50 replay/source receipt failed')
    head=load('cesfl_r50_head',SOURCE/'CES_FL/src/ces_fl_head_ablation_resnet50.py')
    if head.sha(head.__file__)!=HEAD_SHA:raise RuntimeError('ResNet downstream source hash changed')
    git='/home/soft/anaconda3/pkgs/git-2.45.2-pl5340h9abc3c3_1/bin/git'
    commit=subprocess.check_output([git,'-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if subprocess.run([git,'-C',str(SOURCE),'symbolic-ref','-q','HEAD'],stdout=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('Runner checkout must be detached')
    if subprocess.check_output([git,'-C',str(SOURCE),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Runner source tree is dirty')
    arm_key='A' if args.arm=='L_A' else 'D'
    arm_root=ROOT/args.arm
    if arm_root.exists():raise RuntimeError(f'Preserve existing arm output {arm_root}')
    template=read_json(SOURCE/f'CES_FL/proposals/lh_finalization_celebdf_20261104_r1/configs/resnet50_{args.arm}.template.json')
    template['init_sha256']=head.sha(P50_DIR/'best.pt')
    arm_root.mkdir(parents=True)
    template_path=arm_root/f'{args.arm}.input_template_PRIVATE.json'
    template_path.write_text(json.dumps(template,indent=2)+'\n')
    bound_path=arm_root/'config.json'
    head.bind(template_path,bound_path)
    config=json.loads(bound_path.read_text());preflight_dir=arm_root/'preflight'
    head.preflight(config,preflight_dir)
    qualification=None
    if args.arm=='L_H':
        preview_dir=arm_root/'preview';head.preview(config,preflight_dir,preview_dir)
        review=arm_root/'review_reused_PRIVATE.json';qualification_path=arm_root/'qualification.json'
        helper=SOURCE/'CES_FL/proposals/lh_finalization_celebdf_20261104_r1/reuse_procedural_review_if_identical.py'
        subprocess.run([PYTHON,str(helper),'--new-preview',str(preview_dir),'--derived-review',str(review),
                        '--qualification',str(qualification_path)],check=True,cwd=SOURCE)
        qualification=str(qualification_path)
    budget=load('cesfl_budget_direct',Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/allocated_stage.py'))
    sid=f'lh_resnet50_{args.arm}_s17_20261104'
    gpu=card_by_uuid(args.gpu_uuid)
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    import torch
    lease_dir=LOCKS/f"gpu{gpu['index']}";lease=lease_dir/f'task_{args.arm}_r50_s17.json'
    with pool_mutex():
        # Recheck card and global lease state while holding the shared-pool mutex.
        gpu=card_by_uuid(args.gpu_uuid)
        if lease.exists():raise RuntimeError('This arm already has a GPU lease')
        if lease_dir.exists():
            owner=lease_dir/'pool_owner.json'
            if not owner.exists() or not read_json(owner).get('shared'):
                raise RuntimeError('GPU is not an authorized shared pool-v2 card')
            active=[]
            for path in lease_dir.glob('task_*.json'):
                entry=read_json(path);pid=entry.get('pid')
                if isinstance(pid,int) and Path(f'/proc/{pid}').exists():active.append(entry)
            if active:raise RuntimeError(f'GPU already leased: {active}')
        else:
            lease_dir.mkdir();write_json(lease_dir/'pool_owner.json',{'shared':True,'v':2,'pid':0,'wid':'pool_v2_shared'})
        write_json(lease,{'name':args.arm,'wid':args.arm,'pid':os.getpid(),'pgid':os.getpgrp(),
                          'cls':'writer','need':24000,'time':time.time(),'device_uuid':gpu['uuid'],'commit':commit})
    try:
        reserve(budget,sid,gpu,head,commit)
    except Exception:
        with pool_mutex():
            if lease.exists() and read_json(lease).get('pid')==os.getpid():lease.unlink()
        raise
    start=time.monotonic();stop=threading.Event();state={'status':'FAILED'}
    def heartbeat():
        while not stop.wait(20):
            with budget.ledger() as d:
                row=next(s for s in d['gpu_sessions'] if s['id']==sid);row['elapsed_seconds']=time.monotonic()-start
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        verify_cuda_uuid(torch,gpu['uuid'])
        smoke_receipt=ROOT/'P50'/'checks'/'synthetic_downstream'/'SMOKE_RESULT.json'
        if not smoke_receipt.is_file():raise RuntimeError('Current source-bound CUDA smoke receipt missing')
        smoke=read_json(smoke_receipt)
        if smoke.get('status')!='PASS' or smoke.get('source_sha256')!=HEAD_SHA or smoke.get('device')!='cuda':
            raise RuntimeError('Source-bound CUDA smoke receipt mismatch')
        drydir=arm_root/'dry_run';head.dry_run(config,preflight_dir,drydir,arm_key,torch.device('cuda'))
        dry_receipt=drydir/'DRY_RUN.json'
        dry=read_json(dry_receipt)
        if dry.get('status')!='PASS' or dry.get('source_sha256')!=HEAD_SHA or dry.get('device')!='cuda':
            raise RuntimeError('Real-input backward receipt mismatch')
        if dry.get('config_digest')!=head.digest(config) or dry.get('arm')!=arm_key or dry.get('optimizer_steps_in_discarded_model')!=1:
            raise RuntimeError('Real-input backward was not this exact arm/config or unexpectedly persisted updates')
        training_result=head.train(config,preflight_dir,arm_root/'training',arm_key,torch.device('cuda'),qualification,
                                   resume=False,budget_hours=RESERVED_HOURS)
        if training_result.get('status')!='COMPLETED_VERIFIED':
            state={'status':'INCOMPLETE','arm':args.arm,'training_result':training_result,
                   'test_evaluated':False}
            raise RuntimeError(f'Arm did not finish all configured epochs and validation replay: {training_result}')
        state={'status':'COMPLETED','arm':args.arm,'selected_checkpoint_sha256':head.sha(arm_root/'training'/'best.pt'),
               'epochs':config['epochs'],'p50_sha256':head.sha(P50_DIR/'best.pt'),'test_evaluated':False}
    finally:
        stop.set();thread.join(timeout=20);elapsed=time.monotonic()-start
        state.update(elapsed_seconds=elapsed,gpu_hours=elapsed/3600,gpu=gpu,host=os.uname().nodename,commit=commit)
        write_json(arm_root/'ALLOCATION_END.json',state)
        with budget.ledger() as d:
            row=next(s for s in d['gpu_sessions'] if s['id']==sid);row.update(state=state['status'],elapsed_seconds=elapsed)
        with pool_mutex():
            if lease.exists() and read_json(lease).get('pid')==os.getpid():lease.unlink()


if __name__=='__main__':main()
