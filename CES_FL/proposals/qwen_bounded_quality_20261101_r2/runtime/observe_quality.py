"""Finite status publication only. Cannot launch, retry, approve, train or cancel."""
import datetime
import fcntl
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/qwen_quality32_20261101T095157Z')
HELPER = Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_recovery_20261031T165803Z/runtime/execute_handoff.py')
STATUS = 'CES_FL/status/' + ROOT.name + '_live.md'
PUBLIC = 'CES_FL/results/' + ROOT.name + '/status.json'


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def observe():
    submission = read(ROOT / 'checks/pilot_SUBMISSION.json')
    progress = read(ROOT / 'checks/pilot_PROGRESS.json')
    failure = read(ROOT / 'checks/pilot_FAILURE.json')
    passed = read(ROOT / 'checks/pilot_PASS.json')
    binding = read(ROOT / 'checks/pilot_BINDING.json')
    profile = read(ROOT / 'checks/PILOT_PROFILE.json')
    if profile:
        profile = {k: v for k, v in profile.items() if k != 'output_records_aggregate'}
    counts, source_labels, complete, total = {}, {}, 0, 0
    for path in (ROOT / 'generation/items').glob('*/record.json'):
        record = read(path)
        if not record:
            continue
        complete += len(record.get('variants', {})) == 7
        for condition in record.get('variants', {}):
            counts[condition] = counts.get(condition, 0) + 1
            label = str(record['source']['label'])
            source_labels[label] = source_labels.get(label, 0) + 1
            total += 1
    liveness = 'UNVERIFIED'
    if submission:
        try:
            p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', submission['host'],
                                'ps -p ' + str(int(submission['pid'])) + ' -o args='],
                               capture_output=True, text=True, timeout=12)
            expected = str(ROOT / 'runtime/direct_worker.py')
            liveness = 'LIVE' if p.returncode == 0 and expected in p.stdout else 'NOT_LIVE'
        except (OSError, subprocess.TimeoutExpired):
            pass
    state = 'COMPLETED_GENERATION_NOT_VALIDATED' if passed else 'FAILED' if failure else (
        (progress or {}).get('phase', 'SUBMITTED_UNVERIFIED'))
    return {'observed_management_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'state': state, 'submission': submission, 'worker_liveness': liveness,
            'progress': progress, 'binding': binding, 'failure': failure,
            'pass_receipt': passed, 'profile': profile, 'complete_sources': complete,
            'planned_sources': 32, 'generated_outputs': total, 'planned_outputs': 224,
            'condition_counts': counts, 'counts_by_original_source_label_not_candidate_ground_truth': source_labels,
            'source_sha256': '3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab',
            'selection_tests': read(ROOT / 'checks/SELECTION_TESTS.json'),
            'training_started': False, 'full_bank_generation_started': False,
            'human_approvals_created': 0, 'forensic_validity_established': False,
            'FF2': 'DO NOT RUN', 'run_root': str(ROOT), 'observer_pid': os.getpid(),
            'observer_scope': '8h maximum; read/publish only; no launch, retry, approval or cancellation'}


def render(d):
    binding = d['binding'] or {}
    return '\n'.join([
        '# Qwen 32-source quality pilot: actual execution snapshot', '',
        'Observed management UTC: **' + d['observed_management_utc'] + '**.', '',
        '**State: ' + d['state'] + '**. Worker liveness: ' + d['worker_liveness'] + '.', '',
        '| Item | Actual status |', '|---|---|',
        '| Completed originals | ' + str(d['complete_sources']) + '/32 |',
        '| Generated diffusion outputs | ' + str(d['generated_outputs']) + '/224 |',
        '| Host / worker PID | ' + str((d['submission'] or {}).get('host')) + ' / ' + str((d['submission'] or {}).get('pid')) + ' |',
        '| Bound physical GPU | ' + str(binding.get('physical_index', 'not bound yet')) + ' |',
        '| Source-selection CPU checks | 6 passed; synthetic stubs do not validate edits |',
        '| Human approval / forensic preservation | Not established for new candidates |',
        '| Training / full bank / test evaluation | Not started |',
        '| FairForge-2 | DO NOT RUN |', '',
        'Expected generation: approximately **4–6 hours once suitable GPU headroom is secured**, plus review/retries. '
        'GPU waiting is additional; the worker gives up after two hours without eligible headroom. No completion time is guaranteed.', '',
        'This is the explicitly approved bounded **quality-only** diagnostic: 32 originals, 16 real and 16 fake, two per label × ids8 cell. '
        'It retains 40 steps, BF16, the pinned Qwen model, seven independent conditions and resize controls. '
        'It is not a reduced-coverage execution of the full 76,139-original study. The full-training coverage guard rejects this subset. '
        'Within-sample contributor proxies are disjoint, but eight proxies overlap VAL under the preserved benchmark split; no identity-disjoint study is claimed.', '',
        'The parent 14-output pilot was reportedly reviewed by the user; individual decisions and substantive forensic-validity evidence remain unrecorded. '
        '“Go ahead” was recorded as permission for this bounded diagnostic, not fabricated per-pair approvals.', '',
        'Generation progress and source labels are not classification results. No fairness improvement or win over Ding is established. '
        'New candidates cannot enter training without actual reviews and a separately justified training protocol.', '',
        'Direct SSH uses the existing shared dispatcher exclusion plus a project lease, at least 53,248 MiB free before loading, '
        'and a 55% allocator cap. Existing processes continue; no PBS allocation or exclusive ownership is claimed.', '',
        'Run root: `' + str(ROOT) + '`. Worker log: `logs/pilot_worker.log`; generation log: `logs/qwen_32_source_quality.log`.', '',
        '[Protocol, operational code and authorization](../proposals/qwen_bounded_quality_20261101_r2/README.md) · '
        '[Machine-readable actual status](../results/' + ROOT.name + '/status.json).', '',
        'This finite observer updates only these status files, normally no more than every ten minutes and on terminal state. '
        'It cannot advance to training, retry failures or validate images. Read the observation timestamp.'
    ]) + '\n'


def main():
    lock = (ROOT / 'observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = importlib.util.spec_from_file_location('status_publish_only', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.ROOT = ROOT
    started, last_publish, last_phase = time.monotonic(), 0, None
    missing = 0
    while time.monotonic() - started < 8 * 3600:
        d = observe()
        missing = missing + 1 if d['worker_liveness'] == 'NOT_LIVE' else 0
        terminal = d['pass_receipt'] is not None or d['failure'] is not None or missing >= 3
        (ROOT / 'STATUS.md').write_text(render(d))
        (ROOT / 'CURRENT_STATUS.json').write_text(json.dumps(d, indent=2) + '\n')
        if terminal or d['state'] != last_phase or time.monotonic() - last_publish >= 600:
            try:
                data_commit = helper.github_file(PUBLIC, json.dumps(d, indent=2) + '\n')
                page_commit = helper.github_file(STATUS, render(d))
                (ROOT / 'last_publication.json').write_text(json.dumps({'data_commit': data_commit, 'status_commit': page_commit,
                    'observed_management_utc': d['observed_management_utc']}, indent=2) + '\n')
                print('GITHUB_STATUS_PUSH', data_commit, page_commit, flush=True)
                last_publish, last_phase = time.monotonic(), d['state']
            except Exception as exc:
                print('PUBLICATION_ERROR', repr(exc), flush=True)
        if terminal:
            print('OBSERVER_FINISHED', d['state'], d['worker_liveness'], flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
