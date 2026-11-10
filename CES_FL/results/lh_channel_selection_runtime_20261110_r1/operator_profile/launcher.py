"""Finite one-GPU diagnostic follow-up; never starts production training."""
from datetime import datetime,timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/lh_channel_selection_20261110_r1_operator_1a6d63d')
CONTROL=Path('/home/sachin.chaudhary/ces_fl_sources/lh-channel-selection-control-20261110')
spec=importlib.util.spec_from_file_location('prior_operator',CONTROL/'controller.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
SESSION='lh_channel_selection_1a6d63d_representative_profile_attempt2_20261110'


def main():
    c.verify()
    source=Path(__file__).resolve().parent;binding=c.read(ROOT/'REPRESENTATIVE_PROFILE_BINDING.json')
    assert c.detached_head(source)==binding['commit']
    for name,h in binding['files'].items():assert c.sha(source/name)==h
    previous=c.read(ROOT/'REPRESENTATIVE_PROFILE_ACCOUNTING.json');base=previous['aggregate_cohort_GPU_seconds']
    assert previous['status']=='FAILED' and base<7200
    budget=c.load(c.BUDGET_HELPER,'profile_budget');pool=c.load(c.POOL_HELPER,'profile_pool')
    remaining=(7200-base)/3600;cap_seconds=720;reserved=False;entry=None;proc=None;started=None
    stop=threading.Event();status='FAILED';elapsed=0.
    with budget.ledger() as d:
        assert not any(s['id']==SESSION for s in d['gpu_sessions'])
        committed=sum(s['reserved_gpu_hours'] if s['state'] in ('QUEUED','RUNNING') else s.get('elapsed_seconds',0)*s['ngpus']/3600 for s in d['gpu_sessions'])
        assert committed+remaining<=d['initial_total_gpu_hours']
        d['gpu_sessions'].append(dict(id=SESSION,state='QUEUED',ngpus=1,reserved_gpu_hours=remaining,elapsed_seconds=0.,
            purpose='Actual schedule diagnostic after conservative single-step cost block; no production training',
            parent_cohort=c.SESSION,prior_cohort_GPU_seconds=base,maximum_this_profile_GPU_seconds=cap_seconds,
            route='user_authorized_direct_ssh_pool_v2',pbs_allocation_claimed=False))
    reserved=True
    def charge(final=False):
        used=time.monotonic()-started if started is not None else 0.
        with budget.ledger() as d:
            s=next(s for s in d['gpu_sessions'] if s['id']==SESSION)
            s.update(state=status if final else 'RUNNING',elapsed_seconds=used,host=os.uname().nodename,controller_pid=os.getpid())
    try:
        with pool.mutex():
            choices=pool.candidates();assert choices,'No eligible idle shared GPU'
            card=choices[0];directory=pool.LOCKS/('gpu'+str(card['index']));directory.mkdir(exist_ok=True)
            if not (directory/'pool_owner.json').exists():pool.write(directory/'pool_owner.json',dict(shared=True,v=2,pid=0,wid='pool_v2_shared'))
            entry=directory/'task_lhchannel_representative_profile.json';assert not entry.exists()
            pool.write(entry,dict(name='lhchannel_representative_profile',wid='lhchannel_representative_profile',pid=os.getpid(),pgid=None,cls='reader',need=16000,time=time.time()))
            started=time.monotonic()
        old=c.read(ROOT/'ALLOCATION_primary.json');allocation=dict(old)
        allocation.update(reservation_id=str(entry),gpu_uuid=card['uuid'],diagnostic_only=True,original_cohort_window_preserved=True)
        allocation_path=ROOT/'ALLOCATION_representative_profile_attempt2.json';c.write(allocation_path,allocation)
        print(json.dumps(dict(event='REPRESENTATIVE_PROFILE_STARTED',card=card,prior_GPU_seconds=base,profile_cap_seconds=cap_seconds)),flush=True)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=card['uuid'],CES_GPU_ALLOCATION_RECEIPT=str(allocation_path),
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        bootstrap="import torch,sys,runpy;torch.cuda.set_per_process_memory_fraction(.18);sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
        with (ROOT/'representative_profile_attempt2.log').open('x') as f:
            proc=subprocess.Popen([c.PYTHON,'-u','-c',bootstrap,str(source/'training_schedule_profile.py'),'--out',str(ROOT/'REPRESENTATIVE_PROFILE_attempt2.json')],cwd=source,env=env,
                stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            with pool.mutex():lease=pool.read(entry);lease['pgid']=proc.pid;pool.write(entry,lease)
            def watchdog():
                while not stop.wait(.25):
                    if time.monotonic()-started>=cap_seconds:
                        if proc.poll() is None:
                            try:os.killpg(proc.pid,signal.SIGTERM)
                            except ProcessLookupError:pass
                        return
            watcher=threading.Thread(target=watchdog,daemon=True);watcher.start();last=0.
            while proc.poll() is None:
                now=time.monotonic()
                if now-started>=cap_seconds+3:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise RuntimeError('Bounded profile deadline reached')
                if now-last>=20:charge();last=now
                time.sleep(.5)
            assert proc.returncode==0,'Profile failed; no automatic retry'
        p=c.read(ROOT/'REPRESENTATIVE_PROFILE_attempt2.json');assert p['status']=='PASS'
        old_profile=c.read(ROOT/'RUNTIME_PROFILE.json');steps=math.ceil(218266/32)*3;estimates={}
        for arm,r in p['arms'].items():
            overhead=r['input_setup_seconds']+r['model_setup_seconds']+3*r['train_real_calibration_seconds']+4*old_profile['VAL_forward_seconds']+60
            estimates[arm]=1.10*(steps*r['conservative_step_seconds']+overhead)
        used=base+(time.monotonic()-started)
        passed=c.cost_gate(estimates,used)
        c.write(ROOT/'REFINED_COST_GATE_attempt2.json',dict(status='PASS' if passed else 'BLOCKED_RUNTIME_FIT',estimates_seconds=estimates,
            prior_and_current_setup_GPU_seconds=used,steps_per_arm=steps,original_aggregate_limit_GPU_seconds=7200,
            representative_profile_sha256=c.sha(ROOT/'REPRESENTATIVE_PROFILE_attempt2.json'),cost_safety='Worst20step block x1.10 plus measured overhead and another1.10 margin',
            single_step_diagnostic_was_not_steady_state=True,production_training_started=False))
        status='COMPLETED_PROFILE_PASS' if passed else 'COMPLETED_PROFILE_RUNTIME_BLOCK'
        print(json.dumps(dict(event=status,estimates_seconds=estimates)),flush=True)
    except BaseException as error:
        c.write(ROOT/'REPRESENTATIVE_PROFILE_FAILURE_attempt2.json',dict(error=str(error),no_automatic_retry=True));print(str(error),flush=True)
    finally:
        stop.set()
        if 'watcher' in locals():watcher.join(timeout=3)
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        if entry is not None:
            with pool.mutex():
                if entry.exists() and pool.read(entry).get('pid')==os.getpid() and not pool.alive(pool.read(entry).get('pgid'),True):entry.unlink()
        elapsed=time.monotonic()-started if started is not None else 0.
        if reserved:charge(final=True)
        c.write(ROOT/'REPRESENTATIVE_PROFILE_ACCOUNTING_attempt2.json',dict(status=status,current_GPU_seconds=elapsed,prior_GPU_seconds=base,
            aggregate_cohort_GPU_seconds=base+elapsed,aggregate_cohort_GPU_hours=(base+elapsed)/3600,limit_GPU_hours=2,lease_released=True))


if __name__=='__main__':main()
