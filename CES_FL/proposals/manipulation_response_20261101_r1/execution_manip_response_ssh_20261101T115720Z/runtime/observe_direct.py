"""Read-only finite observation of explicitly launched SSH milestones; no launches."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z')
HELPER = Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_recovery_20261031T165803Z/runtime/execute_handoff.py')
OLD = Path('/home/sachin.chaudhary/ces_fl_runs/manip_response_20261101T104925Z')
PUBLIC = 'CES_FL/results/' + ROOT.name + '/status.json'
PAGE = 'CES_FL/status/' + ROOT.name + '_live.md'


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def history(path):
    if not path.is_file():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1:
                raise
    return rows


def stage_state(stage):
    submission = read(ROOT / 'checks' / (stage + '_SUBMISSION.json'))
    progress = read(ROOT / 'checks' / (stage + '_PROGRESS.json'))
    passed = read(ROOT / 'checks' / (stage + '_PASS.json'))
    failure = read(ROOT / 'checks' / (stage + '_FAILURE.json'))
    binding = read(ROOT / 'checks' / (stage + '_BINDING.json'))
    state = 'FAILED' if failure else 'COMPLETED' if passed else 'BLOCKED_INPUTS'
    live = 'NOT_APPLICABLE'
    if not failure and not passed and submission:
        try:
            p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', submission['host'],
                                'ps -p ' + str(int(submission['pid'])) + ' -o args='],
                               capture_output=True, text=True, timeout=12)
            live = 'LIVE' if p.returncode == 0 and str(ROOT / 'runtime/direct_mr.py') in p.stdout else 'NOT_LIVE'
        except (OSError, subprocess.TimeoutExpired):
            live = 'UNVERIFIED'
        if live == 'LIVE' and (progress or {}).get('phase') == 'RUNNING':
            state = 'RUNNING'
    return {'status': state, 'submission': submission, 'progress': progress, 'pass': passed,
            'failure': failure, 'binding': binding, 'liveness': live}


def observe():
    stages = {s: stage_state(s) for s in ('smoke', 'real_preflight', 'P', 'bce', 'fair')}
    arms = {}
    for arm in ('P', 'bce', 'fair'):
        actual = stages[arm]
        records = history(ROOT / arm / 'history.jsonl')
        receipt = read(ROOT / arm / 'RUN_VERIFIED.json')
        completed = (ROOT / arm / 'DONE').is_file() and receipt and receipt.get('status') == 'PASS'
        if actual['status'] == 'COMPLETED' and not (completed and len(records) == 50):
            actual = {**actual, 'status': 'BLOCKED_INPUTS', 'completion_issue': 'Missing native P50/arm50 replay or epoch evidence'}
        checkpoint = None
        if completed:
            cache = ROOT / 'checks' / (arm + '_OBSERVED_CHECKPOINT.json')
            checkpoint = read(cache)
            if checkpoint is None:
                digest = sha(ROOT / arm / 'best.pt')
                assert digest == receipt['checkpoint_sha256'], 'Native checkpoint bytes differ from replay receipt'
                checkpoint = {'sha256': digest, 'native_replay_receipt': receipt}
                cache.write_text(json.dumps(checkpoint, indent=2) + '\n')
        log = ROOT / 'logs' / (arm + '.log')
        arms[arm] = {**actual, 'recorded_epochs': len(records), 'budget_epochs': 50,
                     'latest_epoch': records[-1] if records else None,
                     'selected_epochs_recorded': [r['epoch'] for r in records if r.get('selected_this_epoch')],
                     'verified_checkpoint': checkpoint,
                     'actual_log_tail': log.read_text()[-3000:] if log.exists() else None}
    for arm in ('nochange', 'score', 'response'):
        arms[arm] = {'status': 'BLOCKED_INPUTS', 'reason': 'Complete independently reviewed matched manipulation quartet bank, P50, binding and configured per-arm backward checks missing'}
    decode = read(ROOT / 'real_inputs/DECODE_AUDIT.json')
    dry = read(ROOT / 'P_real_dry/RESULT.json')
    smoke = read(ROOT / 'smoke_cuda/RESULT.json')
    return {'observed_management_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_commit': '58773bbef9d9fe79dd313610f694ddeea12c6037', 'run_root': str(ROOT),
            'execution_mode': 'user_authorized_direct_ssh_bounded_shared_gpu', 'pbs_allocation': None,
            'source_or_scientific_configuration_changed': False,
            'route_override': read(ROOT / 'checks/ROUTE_AUTHORIZATION.json'),
            'stages': stages, 'arms': arms, 'cuda_smoke_result': smoke,
            'real_input_audit': decode, 'configured_backward': dry,
            'real_input_log_tail': (ROOT/'logs/real_preflight.log').read_text()[-1800:] if (ROOT/'logs/real_preflight.log').exists() else None,
            'quartet_preparation': read(OLD/'quartet_preparation/AGGREGATE.json'),
            'qwen_job_modified': False, 'test_evaluation_launched': False, 'FF2_launched': False,
            'observer_pid': os.getpid(),
            'observer_scope': '72h maximum; read/publish only; cannot launch, chain, retry, approve or terminate work'}


def render(d):
    lines = ['# Manipulation-response seed17: direct SSH execution', '',
             'Observed management UTC: **' + d['observed_management_utc'] + '**.', '',
             'The user superseded PBS with SSH to gpu01/gpu02. Queued smoke44963 was cancelled **before execution**; '
             'its records are preserved. No PBS allocation or exclusive ownership is claimed. The existing Qwen pilot is untouched.', '',
             'Exact scientific source: `58773bbef9d9fe79dd313610f694ddeea12c6037`. Architecture, loss, config, seed, selection and calibration are unchanged.', '',
             '| Check | Actual state |', '|---|---|',
             '| New CUDA software suite | ' + d['stages']['smoke']['status'] + ' |',
             '| Full original TRAIN/VAL decode/hash + configured backward | ' + d['stages']['real_preflight']['status'] + ' |', '',
             '| Arm | Actual status | Recorded epochs |', '|---|---|---|']
    for arm, v in d['arms'].items():
        lines.append('| ' + arm + ' | ' + v['status'] + ' | ' + str(v.get('recorded_epochs', 0)) + '/50 |')
    lines += ['', 'P/BCE/fair need no quartet bank. All paired arms remain blocked by valid matched manipulations and independent review. '
              'No source-real substitution, Qwen-only quartet, synthetic-fixture bank or fabricated approval is accepted.', '',
              'Each SSH milestone acquires the shared dispatcher exclusion and project lease, protects Qwen, requires **20GiB free headroom**, '
              'and uses a **15% Torch allocator cap** on one verified CUDA UUID. Existing processes continue. Both nodes were inspected; '
              'actual per-stage host, physical GPU and worker/child PIDs are in the machine record. Duplicate jobs are not launched.', '',
              'The synthetic six-group CUDA suite is software evidence, not a demographic/forensic validity or fairness result. '
              'The original-only track uses all76,139 TRAIN originals, validation-only AUC selection and frozen validation-real empirical10% FPR calibration. '
              'Recorded validation metrics are not test results or a win over Ding. No FairForge-2 or test evaluation is attached.', '',
              'Run root: `' + str(ROOT) + '`. Actual logs: `logs/smoke.log`, `logs/real_preflight.log`, `logs/P.log` when those stages execute. '
              'Only explicitly requested single milestones run; this observer cannot advance them.', '',
              '[Actual metrics, PIDs, bindings, hashes and failures](../results/' + ROOT.name + '/status.json) · '
              '[SSH execution record](../proposals/manipulation_response_20261101_r1/execution_' + ROOT.name + '/README.md) · '
              '[Matched quartet blockers](../proposals/manipulation_response_quartets_20261101_r1/README.md).', '',
              'Finite observation lasts at most72h, normally publishing every ten minutes/on state changes. Read the observation timestamp; uninterrupted monitoring is not guaranteed.']
    for name, v in d['stages'].items():
        if v['failure']:
            lines += ['', 'Failure in ' + name + ': `' + v['failure']['exception'] + '`. Subsequent launches are stopped.']
    return '\n'.join(lines) + '\n'


def main():
    lock = (ROOT / 'observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = importlib.util.spec_from_file_location('publication_helpers_only', HELPER)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper); helper.ROOT = ROOT
    started, last, previous, archived = time.monotonic(), 0, None, set()
    while time.monotonic()-started < 72*3600:
        try:
            d = observe()
            (ROOT/'CURRENT_STATUS.json').write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')
            (ROOT/'STATUS.md').write_text(render(d))
            signature = tuple(v['status'] for v in d['stages'].values())
            if signature != previous or time.monotonic()-last >= 600:
                c1 = helper.github_file(PUBLIC, json.dumps(d, indent=2, allow_nan=False)+'\n')
                c2 = helper.github_file(PAGE, render(d))
                for name, state in d['stages'].items():
                    if state['status'] in ('COMPLETED','FAILED') and name not in archived:
                        log=ROOT/'logs'/(name+'.log')
                        # Only the bounded synthetic-software log is published whole.
                        if name=='smoke' and log.is_file():
                            helper.github_file('CES_FL/results/'+ROOT.name+'/smoke_cuda.log',log.read_text())
                        archived.add(name)
                (ROOT/'last_publication.json').write_text(json.dumps({'data_commit':c1,'status_commit':c2,
                    'observed_management_utc':d['observed_management_utc']},indent=2)+'\n')
                print('GITHUB_STATUS_PUSH',c1,c2,flush=True)
                last,previous=time.monotonic(),signature
            if any(v['status']=='FAILED' for v in d['stages'].values()):
                print('OBSERVER_STOPPED_ON_FAILURE_NO_RETRY',flush=True);return
            if all(d['arms'][a]['status']=='COMPLETED' for a in ('P','bce','fair')):
                print('ORIGINAL_TRACK_COMPLETED_PAIRED_TRACK_STILL_GATED',flush=True);return
        except Exception as exc:
            print('OBSERVATION_OR_PUBLICATION_ERROR',repr(exc),flush=True)
        time.sleep(60)


if __name__=='__main__':
    main()
