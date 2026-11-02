"""User-authorized resource expansion only; scientific source/config stay pinned.

Two independent single-GPU controls. A PBS ngpus=2 count reservation is distinct
from physical mapping: when PBS exposes no mapping, use the explicitly authorized
shared pool-v2 route, record that limitation, and never claim exclusive devices.
"""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_worker():
    spec = importlib.util.spec_from_file_location('controls_followup', ROOT / 'followup_stage.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def candidates_for(candidates, allowed, excluded):
    return [c for c in candidates if c['uuid'] in allowed and c['uuid'] not in excluded]


def verify_grant(text):
    if not re.search(r'job_state\s*=\s*R\b', text):
        raise RuntimeError('PBS job is not RUNNING')
    if not re.search(r'Job_Owner\s*=\s*sachin\.chaudhary@', text):
        raise RuntimeError('PBS owner mismatch')
    if not re.search(r'exec_vnode\s*=\s*\(gpu01:[^\n]*ngpus=2', text):
        raise RuntimeError('PBS does not confirm actual gpu01 ngpus=2 grant')


def child(arm, uuid):
    worker = load_worker()
    original_snapshot = worker.direct.pool.snapshot

    def restricted_snapshot(need):
        choices, cards = original_snapshot(need)
        return candidates_for(choices, {uuid}, set()), cards

    worker.direct.pool.snapshot = restricted_snapshot
    sys.argv = [str(ROOT / 'followup_stage.py'), '--dataset', 'ffpp', '--mode', 'arm', '--arm', arm]
    worker.main()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', choices=list('AB'))
    p.add_argument('--uuid')
    args = p.parse_args()
    if args.worker:
        if not args.uuid or not args.uuid.startswith('GPU-'):
            raise RuntimeError('Explicit UUID required')
        child(args.worker, args.uuid)
        return

    worker = load_worker()
    direct = worker.direct
    job = os.environ.get('PBS_JOBID')
    if not job:
        raise RuntimeError('Parent requires PBS allocation')
    started = time.monotonic()
    qstat = subprocess.check_output(['/opt/pbs/bin/qstat', '-f', job], text=True, timeout=30)
    (HERE / 'allocated_qstat_PRIVATE.txt').write_text(qstat)
    verify_grant(qstat)
    if direct.budget.sha(direct.SCRIPT) != direct.SHA:
        raise RuntimeError('Pinned source changed')
    run = worker.root_for('ffpp')
    _, digest = worker.config(run)
    for arm in 'AB':
        if (run / arm).exists() or (ROOT / f'ffpp_{arm}_initial_remote.pid').exists():
            raise RuntimeError('Existing control output/launch record: no duplicate')
        receipt = json.loads((run / f'dry_{arm}/DRY_RUN.json').read_text())
        if (receipt['status'], receipt['source_sha256'], receipt['config_digest'], receipt['device']) != ('PASS', direct.SHA, digest, 'cuda'):
            raise RuntimeError('CUDA backward qualification mismatch')

    excluded = set()
    with direct.budget.ledger() as ledger:
        for session in ledger['gpu_sessions']:
            if session['state'] == 'RUNNING' and session.get('host') == os.uname().nodename:
                excluded.add(session.get('card', {}).get('uuid'))
    # Preserve the existing two-rank Ding baseline's physical devices as well.
    for entry in direct.pool.LOCKS.glob('gpu*/task_ding_ff2_public_s17_20261102T130051Z.json'):
        details = direct.pool.read(entry)
        if direct.pool.alive(details.get('pid')) or direct.pool.alive(details.get('pgid'), True):
            index = int(entry.parent.name[3:])
            uuid = subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
            excluded.add(uuid)

    with direct.pool.mutex():
        choices, cards = direct.pool.snapshot(16000)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    allowed = {c['uuid'] for c in cards.values()}
    scheduler_mapping = bool(visible)
    if visible:
        tokens = visible.split(',')
        allowed = {v if v.startswith('GPU-') else cards[int(v)]['uuid'] for v in tokens}
        if len(allowed) != 2:
            raise RuntimeError('Scheduler visibility does not expose exactly two distinct devices')
    choices = candidates_for(choices, allowed, excluded)
    if len(choices) < 2:
        raise RuntimeError('Fewer than two compatible distinct additional physical GPUs; preserve existing jobs')
    chosen = choices[:2]
    receipt = dict(status='PBS_RESOURCE_GRANT_VERIFIED', pbs_job_id=job, allocated_ngpus=2,
                   allocated_ncpus=12, host=os.uname().nodename, source_sha256=direct.SHA,
                   config_digest=digest, arms=['A', 'B'], chosen_devices=chosen,
                   scheduler_physical_mapping_available=scheduler_mapping,
                   physical_binding_route='scheduler-visible plus pool-v2 leases' if scheduler_mapping else 'explicit user-authorized pool-v2 shared leases; PBS physical mapping unavailable',
                   exclusive_gpus_claimed=False, batch_size_or_scientific_settings_changed=False,
                   test_evaluation=False, prior_c_d_and_ding_preserved=True)
    direct.budget.write(HERE / 'PBS_RESOURCE_GRANT.json', receipt)

    # Count idle allocated GPU slots too. Per-arm task time is counted by the
    # unchanged direct-stage ledger; only the remainder is added here.
    overhead_id = 'ffpp_AB_pbs_idle_overhead'
    direct.reserve(overhead_id, 2.)
    with direct.budget.ledger() as ledger:
        row = next(s for s in ledger['gpu_sessions'] if s['id'] == overhead_id)
        row.update(state='RUNNING', host=os.uname().nodename, controller_pid=os.getpid(),
                   pbs_job_id=job, accounting='unused slots: 2*PBS_elapsed - sum(A/B task-held seconds)',
                   elapsed_seconds=0., physical_ngpus_reserved=2)
    children = []
    failed = None
    try:
        for arm, card in zip('AB', chosen):
            label = f'ffpp_{arm}_initial'
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', arm, '--uuid', card['uuid']]
            with (ROOT / (label + '_remote.log')).open('x') as stream:
                child_process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((arm, child_process))
            with (ROOT / (label + '_remote.pid')).open('x') as stream:
                stream.write(str(child_process.pid) + '\n')
            direct.budget.write(ROOT / (label + '_launch.json'), dict(host=os.uname().nodename,
                    pid=child_process.pid, command=command, pbs_job_id=job,
                    pbs_ngpus_count_verified=True, scheduler_physical_mapping_available=scheduler_mapping,
                    physical_uuid=card['uuid'], scientific_source_sha256=direct.SHA))
        while True:
            with direct.budget.ledger() as ledger:
                arm_seconds = sum(s['elapsed_seconds'] for s in ledger['gpu_sessions']
                                  if s['id'] in ('ffpp_A_initial', 'ffpp_B_initial'))
                row = next(s for s in ledger['gpu_sessions'] if s['id'] == overhead_id)
                row['elapsed_seconds'] = max(0., 2*(time.monotonic()-started)-arm_seconds)
            if all(child_process.poll() is not None for _, child_process in children):
                break
            time.sleep(15)
        codes = {a: c.returncode for a, c in children}
        if any(codes.values()):
            raise RuntimeError('Control worker failure; artifacts preserved: ' + repr(codes))
        receipt.update(status='PBS_WORKERS_EXITED', child_returncodes=codes,
                       arms_verified={a: worker.verified(run, a) for a in 'AB'})
    except BaseException as error:
        failed = repr(error)
        receipt.update(status='FAILED_PRESERVE', error=failed)
        raise
    finally:
        # No worker termination/preemption here. If a sibling is still running,
        # leave this job alive until it exits so PBS does not kill valid work.
        for _, child_process in children:
            if child_process.poll() is None:
                child_process.wait()
        with direct.budget.ledger() as ledger:
            arm_seconds = sum(s['elapsed_seconds'] for s in ledger['gpu_sessions']
                              if s['id'] in ('ffpp_A_initial', 'ffpp_B_initial'))
            row = next(s for s in ledger['gpu_sessions'] if s['id'] == overhead_id)
            row.update(elapsed_seconds=max(0., 2*(time.monotonic()-started)-arm_seconds),
                       state='FAILED' if failed else 'COMPLETED')
        receipt['elapsed_seconds'] = time.monotonic()-started
        receipt['allocated_gpu_hours'] = 2*receipt['elapsed_seconds']/3600
        direct.budget.write(HERE / 'PBS_END.json', receipt)


if __name__ == '__main__':
    main()
