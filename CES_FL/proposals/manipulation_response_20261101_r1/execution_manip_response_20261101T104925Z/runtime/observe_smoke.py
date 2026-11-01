"""Finite observation/publication of the submitted smoke job; cannot launch jobs."""
import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/manip_response_20261101T104925Z')
HELPER = Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_recovery_20261031T165803Z/runtime/execute_handoff.py')
PUBLIC = 'CES_FL/results/' + ROOT.name + '/status.json'
PAGE = 'CES_FL/status/' + ROOT.name + '_live.md'


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def observe():
    submission = read(ROOT / 'checks/smoke_SUBMISSION.json')
    job = submission['job_id']
    p = subprocess.run(['qstat', '-xf', '-F', 'json', job], capture_output=True, text=True, timeout=30)
    if p.returncode:
        scheduler = {'job_state': 'UNVERIFIED', 'query_error': p.stderr.strip()}
    else:
        j = json.loads(p.stdout)['Jobs'][job]
        scheduler = {k: j[k] for k in ('Job_Name', 'job_state', 'queue', 'Exit_status', 'exec_host', 'exec_vnode', 'Resource_List', 'resources_used', 'comment') if k in j}
    passed = read(ROOT / 'checks/smoke_PASS.json')
    failure = read(ROOT / 'checks/smoke_FAILURE.json')
    result = read(ROOT / 'smoke_cuda/RESULT.json')
    if failure or (scheduler.get('job_state') == 'F' and scheduler.get('Exit_status') != 0):
        state = 'FAILED'
    elif passed and result and result.get('status') == 'PASS' and result.get('device') == 'cuda':
        state = 'COMPLETED'
    elif scheduler.get('job_state') == 'R':
        state = 'RUNNING'
    elif scheduler.get('job_state') == 'Q':
        state = 'QUEUED'
    elif scheduler.get('job_state') == 'F':
        state = 'FAILED_MISSING_PASS_RECEIPTS'
    else:
        state = 'UNVERIFIED'
    quartet = read(ROOT / 'quartet_preparation/AGGREGATE.json')
    arms = {'P': {'status': 'BLOCKED_INPUTS', 'reason': 'Allocated new CUDA suite and real-input/configured backward checks must pass first; no P50 launched'},
            'bce': {'status': 'BLOCKED_INPUTS', 'reason': 'Verified shared P50, exact binding and BCE backward check missing'},
            'fair': {'status': 'BLOCKED_INPUTS', 'reason': 'Verified shared P50, exact binding and fair backward check missing'}}
    for arm in ('nochange', 'score', 'response'):
        arms[arm] = {'status': 'BLOCKED_INPUTS', 'reason': 'No validated target manipulation operator/complete independently reviewed matched quartet bank; also requires P50, binding and per-arm backward PASS'}
    return {'observed_management_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_commit': '58773bbef9d9fe79dd313610f694ddeea12c6037',
            'source': '/home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb',
            'run_root': str(ROOT), 'local_preflight': read(ROOT / 'checks/LOCAL_PREFLIGHT.json'),
            'smoke': {'state': state, 'job_id': job, 'scheduler': scheduler,
                      'progress': read(ROOT / 'checks/smoke_PROGRESS.json'),
                      'binding': read(ROOT / 'checks/smoke_BINDING.json'), 'result': result,
                      'pass': passed, 'failure': failure,
                      'actual_log_tail': (ROOT / 'logs/smoke.log').read_text()[-6000:] if (ROOT / 'logs/smoke.log').exists() else None,
                      'allocation_log_tail': (ROOT / 'logs/smoke.pbs.log').read_text()[-6000:] if failure and (ROOT / 'logs/smoke.pbs.log').exists() else None},
            'arms': arms, 'quartet_preparation': quartet, 'new_training_launched': False,
            'qwen_existing_job_modified': False, 'FF2_launched': False, 'test_evaluation_launched': False,
            'observer_pid': os.getpid(), 'observer_scope': '12h maximum; observes this smoke only, stops at completion/failure; cannot launch or chain stages'}


def render(d):
    s = d['smoke']
    lines = ['# Manipulation-response FF++ seed17: actual status', '',
             'Observed management UTC: **' + d['observed_management_utc'] + '**.', '',
             'Exact source: `' + d['source_commit'] + '` in its new detached worktree.', '',
             '**New CUDA smoke: ' + s['state'] + ', job ' + s['job_id'] + '**. Requests ngpus=1, eight CPUs, 48GB RAM on gpu01. '
             'A queued request has no assigned physical GPU. Actual allocation/UUID binding is recorded after dispatch.', '',
             'Scheduler: ' + str(s['scheduler'].get('comment', s['scheduler'].get('job_state'))) + '.', '',
             '| Arm | Actual status | Remaining dependency |', '|---|---|---|']
    for arm, state in d['arms'].items():
        lines.append('| ' + arm + ' | ' + state['status'] + ' | ' + state['reason'] + ' |')
    lines += ['', 'The original-only P/BCE/fair track does **not** require a quartet bank. It can proceed after its own software/input/backward gates. '
              'No entire-handoff unattended executor is installed; this observer cannot start the next milestone.', '',
              'Local verification passed four pinned source/config/test hashes, four external input hashes, five allocation-safety unit tests and shell syntax. '
              'This is not a CUDA pass or real image-decode/backward pass. The full new synthetic CUDA suite remains pending until actual execution.', '',
              'Existing SimSwap code/checkpoints were located and hashed as a candidate manipulation operator. Its shared checkout has detector patches; '
              'no target adapter or valid quartets were executed. Prior drivers contain unrelated dataset/test donor pools and must not be run for this experiment. '
              'A separate TRAIN-only immutable adapter and independent review remain necessary.', '',
              'Qwen32 was balanced by ids8, not gender_race. Only 15 of its 16 real-source candidates exclude recorded VAL contributors, '
              'covering five gender_race groups. Eight source candidates can potentially be reused for the specified quotas after individual review; '
              'eight new eligible real TRAIN sources are proposed to fill missing group/direction slots. No supplementation was launched and no approval was invented.', '',
              'All prospective 16 source and 16 donor candidates exclude recorded VAL contributors and have distinct recorded proxies. '
              'These are prospective inputs, not complete quartets, independently identified people or evidence of fairness improvement.', '',
              'The running Qwen pilot is preserved; its device is explicitly protected by the allocation wrapper. No scheduler priority, existing job or dataset split was changed. '
              'FairForge-2 and test evaluation were not launched. No novelty, SOTA or win over Ding is claimed.', '',
              'Run root: `' + str(ROOT) + '`. Logs: `logs/smoke.pbs.log`, `logs/smoke.log`. '
              'The finite observer publishes observations roughly every ten minutes and on scheduler/terminal changes, for at most twelve hours.', '',
              '[Exact proposal](../proposals/manipulation_response_20261101_r1/README.md) · '
              '[Execution records](../proposals/manipulation_response_20261101_r1/execution_' + ROOT.name + '/README.md) · '
              '[Quartet-preparation plan](../proposals/manipulation_response_quartets_20261101_r1/README.md).']
    return '\n'.join(lines) + '\n'


def main():
    lock = (ROOT / 'observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = importlib.util.spec_from_file_location('mr_publication_only', HELPER)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper); helper.ROOT = ROOT
    started, last, old_state = time.monotonic(), 0, None
    while time.monotonic() - started < 12 * 3600:
        try:
            d = observe()
            (ROOT / 'CURRENT_STATUS.json').write_text(json.dumps(d, indent=2) + '\n')
            (ROOT / 'STATUS.md').write_text(render(d))
            state = d['smoke']['state']
            terminal = state in ('COMPLETED', 'FAILED', 'FAILED_MISSING_PASS_RECEIPTS')
            if state != old_state or terminal or time.monotonic() - last >= 600:
                c1 = helper.github_file(PUBLIC, json.dumps(d, indent=2) + '\n')
                c2 = helper.github_file(PAGE, render(d))
                if terminal:
                    for name in ('smoke.log', 'smoke.pbs.log'):
                        log = ROOT / 'logs' / name
                        if log.is_file():
                            helper.github_file('CES_FL/results/' + ROOT.name + '/' + name, log.read_text())
                (ROOT / 'last_publication.json').write_text(json.dumps({'data_commit': c1, 'status_commit': c2,
                    'observed_management_utc': d['observed_management_utc']}, indent=2) + '\n')
                print('GITHUB_STATUS_PUSH', c1, c2, flush=True)
                last, old_state = time.monotonic(), state
            if terminal:
                print('SMOKE_OBSERVER_FINISHED_NO_NEXT_STAGE_LAUNCHED', state, flush=True)
                return
        except Exception as exc:
            print('OBSERVATION_OR_PUBLICATION_ERROR', repr(exc), flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
