"""Read-only finite publication of explicit SSH milestones; no launches."""
import datetime
import fcntl
import importlib.util
import json
import os
import time
from pathlib import Path

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z')
PUBLIC = 'CES_FL/results/' + ROOT.name + '/status.json'
STATUS = 'CES_FL/status/' + ROOT.name + '_live.md'
HELPER = Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_recovery_20261031T165803Z/runtime/execute_handoff.py')


def read(path):
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except json.JSONDecodeError:
        return None


def observe():
    stages = {}
    for stage in ('preflight', 'pilot'):
        p = read(ROOT / 'checks' / (stage + '_PASS.json'))
        failure = read(ROOT / 'checks' / (stage + '_FAILURE.json'))
        submit = read(ROOT / 'checks' / (stage + '_SUBMISSION.json'))
        stages[stage] = {'state': 'PASS' if p else 'FAIL' if failure else 'SUBMITTED_UNVERIFIED' if submit else 'NOT_STARTED',
                         'pass': p, 'failure': failure, 'submission': submit,
                         'progress': read(ROOT / 'checks' / (stage + '_PROGRESS.json')),
                         'binding': read(ROOT / 'checks' / (stage + '_BINDING.json'))}
    plan = read(ROOT / 'generation/plan.json')
    inventory = read(ROOT / 'checks/MODEL_BINARY_INVENTORY.json')
    counts = {}
    completed_sources = 0
    for path in (ROOT / 'generation/items').glob('*/record.json'):
        record = read(path)
        if record:
            for condition in record.get('variants', {}):
                counts[condition] = counts.get(condition, 0) + 1
            completed_sources += len(record.get('variants', {})) == 7
    return {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_commit': 'fc4e63b8811c5813afa0d298a8172ba9555b7e40', 'run_root': str(ROOT),
            'source_sha256': '3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab',
            'config_sha256': '56e77211a966ae12a449c47cdaf8e3fc271f149c6031513747813a1d1abfaef0',
            'execution_mode': 'direct_ssh_bounded_shared_gpu', 'pbs_allocation': None,
            'superseded_pbs_job': '44948.mgmt01, cancelled before execution after explicit SSH instruction',
            'operational_recovery': read(ROOT / 'checks/ENVIRONMENT_RECOVERY_REQUEST.json'),
            'resource_recovery': read(ROOT / 'checks/SHARED_REGISTRY_CONTRACT.json'), 'stages': stages, 'cuda_smoke': read(Path('/home/sachin.chaudhary/ces_fl_runs/cf5050_ssh_recovery_20261101T073243Z/smoke_cuda/RESULT.json')),
            'real_data_dry_run': read(Path('/home/sachin.chaudhary/ces_fl_runs/cf5050_ssh_recovery_20261101T073243Z/dry_pretrain/RESULT.json')),
            'generation_plan': {k: plan[k] for k in ('source_count', 'generated_outputs', 'sources_sha256', 'auto_approval')} if plan else None,
            'model_inventory': {k: inventory[k] for k in ('files', 'total_bytes', 'inventory_sha256', 'elapsed_seconds')} if inventory else None,
            'pilot_generated_counts': counts, 'pilot_complete_sources': completed_sources,
            'pilot_profile': read(ROOT / 'checks/PILOT_PROFILE.json'),
            'full_training': 'NOT_STARTED', 'full_generation': 'NOT_STARTED',
            'quality_or_full_bank_approval': 'NOT_GRANTED', 'FF2': 'DO NOT RUN',
            'observer_pid': os.getpid(), 'observer_scope': '24h maximum, read-only; no launch/retry/kill/approval operations'}


def render(d):
    lines = ['# FF++ 50+50: direct SSH checks and two-source pilot', '',
             'Observed UTC: **' + d['observed_utc'] + '**.', '',
             'The user explicitly superseded the queued PBS route with direct SSH. Job44948 was cancelled before execution. '
             'This route uses bounded shared VRAM and claims no PBS allocation or exclusive GPU ownership.', '',
             'Source commit: `' + d['source_commit'] + '`. Scientific source/configuration/input hashes remain pinned.', '',
             '| Milestone | Actual status |', '|---|---|']
    for stage, v in d['stages'].items():
        phase = (v['progress'] or {}).get('phase', '')
        lines.append('| ' + stage + ' | ' + v['state'] + (' / ' + phase if phase else '') + ' |')
    smoke = d['cuda_smoke']
    dry = d['real_data_dry_run']
    lines += ['| CUDA software smoke | ' + ('PASS, ' + str(smoke['check_count']) + ' actual checks on ' + smoke['device'] if smoke else 'PENDING') + ' |',
              '| Real configured batch backward | ' + ('PASS, zero optimizer steps' if dry and dry['status'] == 'PASS' else 'PENDING') + ' |',
              '| Full P / generation / downstream B,N,C | NOT STARTED |',
              '| Human/forensic-validity approval | NOT GRANTED |', '| FairForge-2 comparison | DO NOT RUN |', '',
              'Completed pilot sources: ' + str(d['pilot_complete_sources']) + '/2. Generated condition counts: `' + json.dumps(d['pilot_generated_counts'], sort_keys=True) + '`.', '',
              'The initial SSH wrapper stopped before model execution because PyTorch reports bare UUIDs while NVIDIA '
              'prefixes them withGPU-. The failed attempt is preserved. The operational recovery compares normalized full UUIDs '
              'and passed three guards: actual bare/prefixed equality, rejection of another device, rejection of malformed IDs. '
              'No scientific source, loss, architecture or configuration was changed.', '',
              'The previous environment-recovered pilot hit physical GPU OOM when another reserved task loaded40.3GiB. This retry additionally holds the existing shared dispatcher legacy GPU lock, preventing new pool work on the selected card while existing processes continue. No pool jobs or configuration were altered. The real pilot is limited to two originals, one per class, with seven independent generated conditions each '
              'and geometric controls. No generated image is automatically approved. Full pretraining/generation requires '
              'quality/resource checks; downstream binding also requires actual full per-source reviews and verified P.', '',
              'Training preflight receipts are reused from the verified unchanged aiface run; source/config/input/inventory bytes match. The failed aiface generator lacked QwenImageEditPlusPipeline. This recovery uses the existing Qwen-compatible llada-image environment plus an isolated CPU-import overlay, leaving training in aiface and all science settings unchanged. Runtime, storage and model-binary provenance will be reported from actual pilot receipts. The full plan remains '
              '76,139 sources and532,973diffusion outputs; coverage is not silently reduced. The observer cannot advance stages.', '',
              '**FairForge-2 remains DO NOT RUN.** [Independent unresolved supervisor review](../proposals/ff2_supervisor_review_20261101_r1/REVIEW.md). '
              'The FF++ configuration must not be substituted.', '',
              '[All actual receipts, bindings, failures and measured pilot profile](../results/' + ROOT.name + '/status.json).', '',
              'Cluster root: `' + str(ROOT) + '`. Logs: `logs/preflight_worker.log`, `logs/smoke_cuda.log`, '
              '`logs/generation_plan.log`, `logs/dry_pretrain.log`, `logs/qwen_two_source_pilot.log`.', '',
              'A finite24h observer publishes these already explicitly launched milestones. It cannot launch, retry, cancel, '
              'approve edits, change settings or evaluate test data. Source inventories, face images, weights and per-image '
              'reviews remain private. Read the observation timestamp.']
    for stage, v in d['stages'].items():
        if v['failure']:
            lines += ['', 'Failure in ' + stage + ': `' + v['failure']['exception'] + '`. Later milestones remain stopped.']
    return '\n'.join(lines) + '\n'


def main():
    lock = (ROOT / 'observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = importlib.util.spec_from_file_location('publish_helpers', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.ROOT = ROOT
    started, previous, last = time.monotonic(), None, 0
    while time.monotonic() - started < 86400:
        d = observe()
        (ROOT / 'STATUS.md').write_text(render(d))
        (ROOT / 'CURRENT_STATUS.json').write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')
        signature = json.dumps({k: v for k, v in d.items() if k != 'observed_utc'}, sort_keys=True)
        if signature != previous or time.monotonic() - last > 1800:
            try:
                c1 = helper.github_file(PUBLIC, json.dumps(d, indent=2, allow_nan=False) + '\n')
                c2 = helper.github_file(STATUS, render(d))
                (ROOT / 'last_publication.json').write_text(json.dumps({'data_commit': c1, 'status_commit': c2,
                     'utc': d['observed_utc']}, indent=2) + '\n')
                print('GITHUB_STATUS_PUSH', c1, c2, flush=True)
                previous, last = signature, time.monotonic()
            except Exception as exc:
                print('PUBLICATION_ERROR', repr(exc), flush=True)
        if d['stages']['preflight']['failure'] or d['stages']['pilot']['failure'] or d['stages']['pilot']['pass']:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
