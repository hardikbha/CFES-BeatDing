"""Finite read-only observation/publication; cannot submit or run milestones."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/cf5050_20261101T065603Z')
SOURCE = Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-cf5050-fc4e63b')
HELPER = Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_recovery_20261031T165803Z/runtime/execute_handoff.py')
PUBLIC_STATUS = 'CES_FL/status/' + ROOT.name + '_live.md'
PUBLIC_JSON = 'CES_FL/results/' + ROOT.name + '/status.json'


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def observe():
    submitted = read(ROOT / 'checks/SUBMISSION.json')
    jobid = submitted.get('job_id') if submitted else None
    scheduler = None
    if jobid:
        p = subprocess.run(['/opt/pbs/bin/qstat', '-x', '-f', '-F', 'json', jobid],
                           text=True, capture_output=True, timeout=20)
        if p.returncode == 0:
            jobs = json.loads(p.stdout)['Jobs']
            item = jobs.get(jobid) or next(iter(jobs.values()))
            fields = ('Job_Name', 'job_state', 'queue', 'Resource_List', 'comment', 'exec_vnode',
                      'Exit_status', 'stime', 'obittime', 'resources_used')
            scheduler = {k: item[k] for k in fields if k in item}
        else:
            scheduler = {'query_exit': p.returncode, 'query_error': p.stderr.strip(), 'job_state': 'UNVERIFIED'}
    passed = read(ROOT / 'checks/STAGE1_PASS.json')
    failed = read(ROOT / 'checks/STAGE1_FAILURE.json')
    state = 'PASS' if passed else 'FAIL_OR_BLOCKED' if failed else 'PENDING'
    if state == 'PENDING' and scheduler:
        state = {'Q': 'QUEUED', 'R': 'RUNNING_MILESTONE1', 'F': 'EXITED_WITHOUT_PASS_RECEIPT'}.get(scheduler.get('job_state'), 'SCHEDULER_UNVERIFIED')
    phases = {}
    for name in ['smoke_cuda', 'generation_plan', 'dry_pretrain']:
        exit_record = read(ROOT / 'checks' / (name + '_EXIT.json'))
        phases[name] = {'state': ('PASS' if exit_record['exit_code'] == 0 else 'FAIL') if exit_record else
                       'LOG_PRESENT_UNVERIFIED' if (ROOT / 'logs' / (name + '.log')).exists() else 'NOT_STARTED',
                       'exit': exit_record, 'log': str(ROOT / 'logs' / (name + '.log'))}
    return {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'job_id': jobid,
            'source_commit': 'fc4e63b8811c5813afa0d298a8172ba9555b7e40', 'source': str(SOURCE),
            'run_root': str(ROOT), 'local_hash_preflight': read(ROOT / 'checks/LOCAL_HASH_PREFLIGHT.json'),
            'scheduler': scheduler, 'milestone1_state': state, 'phases': phases,
            'allocation': read(ROOT / 'checks/ALLOCATION.json'), 'pass': passed, 'failure': failed,
            'qwen_pilot': 'NOT_STARTED', 'full_generation': 'NOT_STARTED', 'full_training': 'NOT_STARTED',
            'downstream_B_N_C': 'BLOCKED: full per-source reviewed bank and verified P checkpoint required',
            'fairforge2': 'DO NOT RUN: independent supervisor blockers remain',
            'observer_scope': '24h maximum; reads and publishes milestone1 only; cannot launch/retry/cancel/change science'}


def render(d):
    local = d['local_hash_preflight']
    lines = ['# FF++ counterfactual 50+50 — actual operator status', '',
             'Observed UTC: **' + d['observed_utc'] + '**.', '',
             '## FF++ study', '',
             f'Source commit `{d["source_commit"]}`; detached checkout `{SOURCE}`.', '',
             f'PBS job **{d["job_id"]}**: **{d["milestone1_state"]}**. Exactly one GPU requested; actual CUDA binding is a runtime gate.', '',
             '| Check or milestone | Actual status |', '|---|---|',
             '| Source/config and four declared input hashes | PASS |']
    for name, value in d['phases'].items():
        lines.append('| ' + name + ' | ' + value['state'] + ' |')
    lines += ['| Two-source real Qwen pilot | NOT STARTED |', '| Full FF++ pretraining and generation | NOT STARTED |',
              '| Full-bank human/source validity review | NOT COMPLETE |', '| Downstream B/N/C binding/training | BLOCKED |', '',
              'Scheduler observation: `' + json.dumps(d['scheduler'], sort_keys=True) + '`.', '',
              'The queued job runs only CUDA synthetic software smoke, the original-image hash inventory/plan, and one '
              'real configured batch backward without an optimizer step. It cannot launch the generator or full training. '
              'If the allocation lacks an actual scheduler CUDA/PBS_GPUFILE device binding, it stops before CUDA execution; '
              'an idle physical GPU is not substituted.', '',
              'Target CUDA software and real-image byte inventory claims require their actual receipts. '
              'The upstream seven-check CPU receipt is not a target CUDA receipt. Current target PyTorch is ' +
              local['environment_versions']['torch'] + '; the upstream CPU receipt used 2.15.0.', '',
              'Cached Qwen snapshot: 35 files, 57,720,463,453 logical bytes; metadata inventory only. Its binary content hashes '
              'must be recorded at the allocated real pilot. Runtime, peak VRAM, output storage, image quality and forensic '
              'preservation remain unmeasured for this exact generator path.', '',
              'Full inventory remains 76,139 originals × 7 diffusion outputs = 532,973 outputs plus 76,139 geometric controls. '
              'Coverage is not reduced to bypass cost or validity failures. No approvals were fabricated or copied from another pool.', '',
              'The next milestone is a separate two-source pilot only after milestone 1 passes. Full pretraining/generation '
              'requires pilot quality and resource checks; downstream training additionally requires actual full coverage '
              'review and the verified P checkpoint. No test evaluation is attached.', '',
              '## FairForge-2 study', '',
              '**DO NOT RUN.** The independent supervisor review remains binding. Unresolved: exact public-split baseline/downstream '
              'specifications; tested class-conditioned Ding variants; common two-logit score adapter; reviewed paired controls; '
              'per-group/paired uncertainty; group/label mapping and actual launch checks. FF++ 50+50 does not close those items.', '',
              '[Exact independent review](../proposals/ff2_supervisor_review_20261101_r1/REVIEW.md). '
              '[FF++ source proposal and gates](../proposals/counterfactual_5050_20261101_r1/GPU_OPERATOR_PROMPT.txt).', '',
              '[Aggregate check/job evidence](../results/' + ROOT.name + '/status.json).', '',
              'Run root: `' + str(ROOT) + '`. PBS log: `' + str(ROOT / 'logs/stage1.pbs.log') + '`.', '',
              'A finite read-only 24h observer publishes these existing milestone 1 artifacts. It never submits, retries, cancels '
              'or advances an experiment. Check the observation time; publication may fail or stop. Full/private images, weights '
              'and row-level review records stay outside GitHub.']
    if d['failure']:
        lines += ['', 'Failure: `' + d['failure']['exception'] + '`. Later milestones remain stopped.']
    return '\n'.join(lines) + '\n'


def main(publish=False, once=False):
    lock = (ROOT / 'observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    helper = None
    if publish:
        spec = importlib.util.spec_from_file_location('publication_helpers', HELPER)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        helper.ROOT = ROOT
    started = time.monotonic()
    previous, last_publication = None, 0
    write(ROOT / 'observer.json', {'pid': os.getpid(), 'read_only': True, 'max_hours': 24,
                                 'helper_sha256': hashlib.sha256(HELPER.read_bytes()).hexdigest(),
                                 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    while time.monotonic() - started < 24 * 3600:
        data = observe()
        write(ROOT / 'CURRENT_STATUS.json', data)
        (ROOT / 'STATUS.md').write_text(render(data))
        signature = json.dumps({k: v for k, v in data.items() if k != 'observed_utc'}, sort_keys=True)
        if publish and (signature != previous or time.monotonic() - last_publication > 1800):
            try:
                c1 = helper.github_file(PUBLIC_JSON, json.dumps(data, indent=2, allow_nan=False) + '\n')
                c2 = helper.github_file(PUBLIC_STATUS, render(data))
                write(ROOT / 'last_publication.json', {'status_commit': c2, 'aggregate_commit': c1,
                                                      'utc': data['observed_utc']})
                print('GITHUB_STATUS_PUSH', c1, c2, flush=True)
                previous, last_publication = signature, time.monotonic()
            except Exception as exc:
                print('PUBLICATION_ERROR', repr(exc), flush=True)
        terminal = data['pass'] or data['failure'] or data['milestone1_state'] == 'EXITED_WITHOUT_PASS_RECEIPT'
        if once or terminal:
            write(ROOT / 'observer_finished.json', {'utc': data['observed_utc'], 'state': data['milestone1_state'],
                                                  'read_only': True, 'published': bool(publish)})
            return
        time.sleep(30)
    write(ROOT / 'observer_finished.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                          'state': '24H_OBSERVATION_LIMIT', 'read_only': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--publish', action='store_true')
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    main(publish=a.publish, once=a.once)
