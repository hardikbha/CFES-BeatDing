#!/usr/bin/env python3
"""Source-bound direct shared-pool runner for Ding-10 controlled adaptation.

Trains only the separately named connected-gradient 10-epoch adaptation.
No TEST access is performed here. The direct pool lease is shared/memory-aware;
it never kills or changes another process.
"""
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

PYTHON='/home/sachin.chaudhary/.conda/envs/aiface/bin/python'
SOURCE=Path(__file__).resolve().parents[2]
MODEL_REL='CES_FL/src/ding10_connected_control.py'
CONFIG_REL='CES_FL/proposals/lh_finalization_celebdf_20261104_r1/configs/ding10_connected.json'
MODEL_SHA='9d480b3bf018d7fff46c840044011122b953b456df04d154f42f173ceeace324'
CONFIG_SHA='3e668b4d29a2f5c369adb39e5ed5319775c248f7fc93c824c40cb8b89c179673'
LEDGER_PATH=Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/allocated_stage.py')
POOL_HELPER=Path('/home/sachin.chaudhary/ces_fl_sources/ding_ff2_public_s17_20261102T130051Z/CES_FL/scripts/run_ding_ff2_ssh.py')
POOL_HELPER_SHA='630bc1811f81a4f9021d84c76ab65aff976e1ffcf93e5faa67a7baac4f547a93'
DIRECT_STAGE=Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/direct_stage.py')
DIRECT_STAGE_SHA='79813d06e563ef7b72b44511b13f044e9cc1c40952fb81f3e69dc3f18dfdc5f8'
POOL_NEED_MIB=16000
MEMORY_FRACTION=0.16
RESERVED_GPU_HOURS=6.0


def sha(path):
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


@contextlib.contextmanager
def pool_mutex(pool):
    p=pool.LOCKS/'.pool_mutex';deadline=time.monotonic()+30
    while True:
        try:p.mkdir();break
        except FileExistsError:
            if time.monotonic()>deadline:raise RuntimeError('Shared GPU pool mutex busy')
            time.sleep(.2)
    try:yield
    finally:p.rmdir()


def current_utilization():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu',
                                 '--format=csv,noheader,nounits'],text=True)
    return {int(r[0]):int(r[1]) for r in csv.reader(out.splitlines())}


def select_and_claim(pool,expected_uuid,sid):
    pool.LOCKS.mkdir(exist_ok=True)
    with pool_mutex(pool):
        candidates,_=pool.snapshot(POOL_NEED_MIB)
        utils=current_utilization()
        candidates.sort(key=lambda c:(utils[c['index']]>20,utils[c['index']],-c['available_mib']))
        matches=[c for c in candidates if c['uuid']==expected_uuid]
        if len(matches)!=1:
            raise RuntimeError(f'Requested shared GPU is not a compatible direct-pool candidate: {expected_uuid}; candidates={candidates}')
        card=matches[0]
        if utils[card['index']]>20:
            raise RuntimeError(f'Shared GPU utilization above20%: {card}, util={utils[card["index"]]}')
        directory=pool.LOCKS/f"gpu{card['index']}"
        if not directory.exists():
            directory.mkdir();pool.write(directory/'pool_owner.json',{'shared':True,'v':2,'pid':0,'wid':'pool_v2_shared'})
        else:
            owner=directory/'pool_owner.json'
            if not owner.exists() or not pool.read(owner).get('shared'):
                raise RuntimeError('Target physical GPU is not registered as shared pool-v2')
        existing=[]
        for f in directory.glob('task_*.json'):
            entry=pool.read(f)
            if pool.alive(entry.get('pid')) or pool.alive(entry.get('pgid'),True):existing.append(entry)
        if existing:raise RuntimeError(f'Target shared GPU already has a direct lease: {existing}')
        lease=directory/f'task_{sid}.json'
        if lease.exists():raise RuntimeError('Duplicate shared lease')
        pool.write(lease,{'name':sid,'wid':sid,'pid':os.getpid(),'pgid':os.getpgrp(),
                          'cls':'writer','need':POOL_NEED_MIB,'time':pool.stamp(),
                          'device_uuid':card['uuid'],'scientific_commit':None})
        card['utilization_at_claim']=utils[card['index']]
        card['active_compute_apps_at_claim']=pool.snapshot(POOL_NEED_MIB)[1].get(str(card['index']),{})
        return card,lease


def reserve(budget,sid,card,commit,config_sha,log):
    with budget.ledger() as d:
        if any(x['id']==sid for x in d['gpu_sessions']):raise RuntimeError('Duplicate Ding-10 budget reservation')
        committed=sum(x['reserved_gpu_hours'] if x['state'] in ('QUEUED','RUNNING')
                      else x.get('elapsed_seconds',0)*x['ngpus']/3600 for x in d['gpu_sessions'])
        if committed+RESERVED_GPU_HOURS>d['initial_total_gpu_hours']:
            raise RuntimeError('Ding-10 exceeds the current authorized study budget')
        d['gpu_sessions'].append({'id':sid,'ngpus':1,'reserved_gpu_hours':RESERVED_GPU_HOURS,'elapsed_seconds':0.,
            'state':'RUNNING','route':'user_authorized_direct_ssh_shared_pool_v2','pbs_allocation_claimed':False,
            'host':os.uname().nodename,'controller_pid':os.getpid(),'card':card,
            'scientific_commit':commit,'scientific_source_sha256':MODEL_SHA,'config_sha256':config_sha,
            'purpose':'Ding SFD+GDA connected-gradient 10-epoch controlled adaptation; no TEST in training',
            'log':str(log)})


def main():
    parser=__import__('argparse').ArgumentParser()
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--attempt-id',required=True)
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--budget-hours',type=float,default=RESERVED_GPU_HOURS)
    args=parser.parse_args()
    if args.budget_hours!=RESERVED_GPU_HOURS:raise ValueError('Ding-10 execution budget is fixed at six GPU-hours')
    if os.uname().nodename.split('.')[0] not in ('gpu01','gpu02'):raise RuntimeError('Direct SSH must land on gpu01/gpu02')
    if not args.gpu_uuid.startswith('GPU-') or not args.attempt_id.replace('_','').replace('-','').isalnum():
        raise ValueError('Invalid physical UUID or attempt identifier')
    git='/home/soft/anaconda3/pkgs/git-2.45.2-pl5340h9abc3c3_1/bin/git'
    commit=subprocess.check_output([git,'-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if subprocess.run([git,'-C',str(SOURCE),'symbolic-ref','-q','HEAD'],stdout=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('Ding-10 executor checkout must be detached')
    if subprocess.check_output([git,'-C',str(SOURCE),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Ding-10 source checkout is dirty')
    model_path=SOURCE/MODEL_REL;config_path=SOURCE/CONFIG_REL
    if sha(model_path)!=MODEL_SHA or sha(config_path)!=CONFIG_SHA:raise RuntimeError('Pinned Ding-10 source/config hash mismatch')
    root=Path(args.run_root).resolve()
    if root.exists():raise RuntimeError(f'Preserve existing Ding-10 output root {root}')
    root.mkdir(parents=True)
    run_root=root/'training'
    sid=f'ding10_connected_s17_20261104_{args.attempt_id}'
    preflight_out=root/'preflight'
    # CPU preflight verifies all TRAIN/VAL inputs and native membership before claiming GPU.
    preflight_log=root/'preflight.log'
    with preflight_log.open('x') as stream:
        command=[PYTHON,'-u',str(model_path),'preflight','--config',str(config_path),'--out',str(preflight_out)]
        stream.write(json.dumps({'command':command,'source_sha256':MODEL_SHA,'config_sha256':CONFIG_SHA})+'\n');stream.flush()
        subprocess.run(command,cwd=SOURCE,stdout=stream,stderr=subprocess.STDOUT,check=True)
    pre=read_json(preflight_out/'PREFLIGHT.json');config=read_json(config_path)
    if pre.get('status')!='PASS' or pre.get('test_read') is not False or pre.get('train')!=76139 or pre.get('val')!=25386:
        raise RuntimeError(f'Fresh Ding-10 CPU input preflight failed: {pre}')
    if (pre.get('audit',{}).get('native_Ding_membership',{}).get('path_label_group_exact') is not True
            or pre.get('audit',{}).get('native_Ding_membership',{}).get('rows')!=76139):
        raise RuntimeError('Fresh Ding-10 preflight checkpoint/SFD/native-row provenance mismatch')
    channels=[int(x) for x in Path(config['sfd_channels']).read_text().split()]
    if pre.get('channels')!=channels:raise RuntimeError('Fresh preflight SFD channel list changed')
    write_json(root/'PREFLIGHT_BOUND.json',{'status':'PASS','source_sha256':MODEL_SHA,
        'config_sha256':CONFIG_SHA,'preflight_sha256':sha(preflight_out/'PREFLIGHT.json'),
        'train_count':pre['train'],'val_count':pre['val'],
        'base_checkpoint_sha256':config['input_hashes']['base_checkpoint'],
        'sfd_channels_sha256':config['input_hashes']['sfd_channels'],
        'native_membership':pre['audit']['native_Ding_membership'],'channels':channels,'test_read':False})
    budget=load('ding10_budget',LEDGER_PATH)
    if sha(DIRECT_STAGE)!=DIRECT_STAGE_SHA:raise RuntimeError('Shared direct-stage lease helper changed')
    pool_helper=load('ding10_pool',POOL_HELPER)
    if sha(POOL_HELPER)!=POOL_HELPER_SHA:raise RuntimeError('Shared pool allocator hash changed')
    card,lease=select_and_claim(pool_helper,args.gpu_uuid,sid)
    try:reserve(budget,sid,card,commit,CONFIG_SHA,root/'allocation.log')
    except Exception:
        with pool_mutex(pool_helper):
            if lease.exists() and pool_helper.read(lease).get('pid')==os.getpid():lease.unlink()
        raise
    start=time.monotonic();stop=threading.Event();state={'status':'RUNNING','completed_stages':[]}
    def heartbeat():
        while not stop.wait(20):
            with budget.ledger() as d:
                row=next(x for x in d['gpu_sessions'] if x['id']==sid);row['elapsed_seconds']=time.monotonic()-start
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=card['uuid'],CES_EXPECTED_GPU_UUID=card['uuid'],
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    bootstrap=("import csv,json,os,runpy,subprocess,sys,torch; "
        "e=os.environ['CES_EXPECTED_GPU_UUID']; "
        "assert torch.cuda.is_available() and torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0); "
        "torch.cuda.set_per_process_memory_fraction(.16,0); torch.empty(1,device='cuda'); "
        "r=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines())); "
        "mine=[x for x in r if len(x)>=2 and x[1].strip()==str(os.getpid())]; "
        "assert any(x[0].strip()==e for x in mine), (e,mine); "
        "print(json.dumps({'event':'CUDA_BINDING_PASS','uuid':e,'device':torch.cuda.get_device_name(0),'rows':mine}),flush=True); "
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')")
    def run_stage(name,cli_args,timeout):
        log=root/f'{name}.log'
        if log.exists():raise ValueError(f'Preserve existing stage log {log}')
        command=[PYTHON,'-u','-c',bootstrap,str(model_path),*cli_args]
        with log.open('x') as stream:
            stream.write(json.dumps({'command':command,'cuda_visible_devices':card['uuid'],
                'source_sha256':MODEL_SHA,'config_sha256':CONFIG_SHA,'shared_gpu_memory_fraction':MEMORY_FRACTION})+'\n');stream.flush()
            child=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            with pool_helper.mutex():
                entry=pool_helper.read(lease);entry['pgid']=child.pid;pool_helper.write(lease,entry)
            with budget.ledger() as d:
                row=next(x for x in d['gpu_sessions'] if x['id']==sid);row['child_pid']=child.pid;row['log']=str(log)
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise RuntimeError(f'{name} exceeded its bounded execution timeout; outputs preserved')
            if code!=0:raise RuntimeError(f'{name} failed with code {code}; inspect {log}')
        return log
    try:
        # One short CUDA process records the actual device mapping before any model work.
        binding_bootstrap=("import csv,json,os,subprocess,torch; e=os.environ['CES_EXPECTED_GPU_UUID']; "
            "assert torch.cuda.is_available() and torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0); "
            "torch.cuda.set_per_process_memory_fraction(.16,0); torch.empty(1,device='cuda'); "
            "r=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines())); "
            "mine=[x for x in r if len(x)>=2 and x[1].strip()==str(os.getpid())]; assert any(x[0].strip()==e for x in mine),(e,mine); "
            "print(json.dumps({'status':'PASS','expected_uuid':e,'device':torch.cuda.get_device_name(0),'rows':mine}),flush=True)")
        binding_log=root/'CUDA_BINDING.log'
        with binding_log.open('x') as stream:
            stream.write(json.dumps({'uuid':card['uuid'],'shared_device':True,'memory_fraction':MEMORY_FRACTION})+'\n');stream.flush()
            subprocess.run([PYTHON,'-u','-c',binding_bootstrap],cwd=SOURCE,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        binding_text=binding_log.read_text()
        if '"status": "PASS"' not in binding_text and '"status":"PASS"' not in binding_text:
            raise RuntimeError('CUDA binding receipt did not contain PASS')
        state['completed_stages'].append('cuda_binding')
        run_stage('cuda_smoke',['smoke','--config',str(config_path),'--out',str(root/'cuda_smoke'),'--device','cuda'],timeout=900)
        smoke=read_json(root/'cuda_smoke'/'SMOKE.json')
        if (smoke.get('status')!='PASS' or smoke.get('source_sha256')!=MODEL_SHA or smoke.get('device')!='cuda'
                or smoke.get('test_read') is not False or smoke.get('lambda_update_delta_norm',0)<=0
                or smoke.get('persistent_optimizer_updates')!=0):
            raise RuntimeError(f'Fresh Ding-10 CUDA smoke failed qualification: {smoke}')
        state['completed_stages'].append('cuda_smoke')
        run_stage('cuda_real_train_backward',['dry-run','--config',str(config_path),'--out',str(root/'cuda_real_dryrun'),'--device','cuda'],timeout=900)
        dry=read_json(root/'cuda_real_dryrun'/'DRY_RUN.json')
        if (dry.get('status')!='PASS' or dry.get('source_sha256')!=MODEL_SHA or dry.get('device')!='cuda'
                or dry.get('test_read') is not False or dry.get('real_input_batch_size')!=32
                or dry.get('persistent_optimizer_updates')!=0 or dry.get('gradient_connected') is not True):
            raise RuntimeError(f'Fresh Ding-10 real TRAIN backward failed qualification: {dry}')
        state['completed_stages'].append('cuda_real_train_backward')
        training_log=run_stage('training',['train','--config',str(config_path),'--out',str(run_root),'--device','cuda'],timeout=5.5*3600)
        receipt=read_json(run_root/'RUN_VERIFIED.json')
        if (not (run_root/'DONE').is_file() or receipt.get('status')!='PASS'
                or receipt.get('test_read') is not False or receipt.get('completed_epochs')!=10
                or receipt.get('validation_replayed') is not True):
            raise RuntimeError(f'Ding-10 training did not complete/replay correctly: {receipt}')
        state.update(status='COMPLETED',selected_checkpoint_sha256=receipt['checkpoint_sha256'],
            selected_epoch=receipt['selected_epoch'],selected_validation_auc=receipt['selected_validation_auc'],
            training_log=str(training_log),test_evaluated=False)
    except Exception:
        if state.get('status')=='RUNNING':state['status']='FAILED_OR_INCOMPLETE'
        raise
    finally:
        stop.set();thread.join(timeout=20);elapsed=time.monotonic()-start
        state.update(elapsed_seconds=elapsed,gpu_hours=elapsed/3600,device=card,host=os.uname().nodename,
                     executor_commit=commit,test_evaluated=state.get('test_evaluated',False))
        write_json(root/'ALLOCATION_END.json',state)
        with budget.ledger() as d:
            row=next(x for x in d['gpu_sessions'] if x['id']==sid);row.update(state=state['status'],elapsed_seconds=elapsed)
        with pool_helper.mutex():
            if lease.exists() and pool_helper.read(lease).get('pid')==os.getpid():lease.unlink()


if __name__=='__main__':main()
