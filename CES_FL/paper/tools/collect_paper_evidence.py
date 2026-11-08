"""Snapshot existing aggregate evidence only; no inference, training, or private scores.

Cluster-only collection. The public draft builds from the resulting snapshot without
cluster access. An existing destination is refused to preserve dated observations.
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def collect(root, out, repo):
    if out.exists():
        raise FileExistsError('Preserve existing snapshot: ' + str(out))
    out.mkdir(parents=True)
    tests = root / 'ff2_test_eval_20261108'
    runs = root / 'ff2_rfa_20261107'
    manifest_sha = 'dd5eb9ef070f658fe6b7f744c47f73e25ce7c53f948bc272f4f16c69a37d4099'
    checks, test_receipts, configs, val, epochs, native = [], {}, {}, {}, [], {}
    for directory in sorted((tests / 'out').iterdir()):
        receipt = read(directory / 'TEST_EVALUATED.json')
        metrics = read(directory / 'metrics.json')
        ck = Path(receipt['checkpoint'])
        actual = sha(ck)
        assert actual == receipt['checkpoint_sha256'] == metrics['checkpoint_sha256'], directory
        assert receipt['manifest_sha256'] == metrics['manifest_sha256'] == manifest_sha, directory
        assert receipt['status'] == 'TEST_EVALUATED'
        # Aggregate reports only: deliberately never open test_scores_PRIVATE.json.
        report = metrics['calibrated']
        assert report['overall']['n'] == 87564
        assert report['overall']['n_real'] == 5441
        assert report['overall']['n_fake'] == 82123
        threshold = metrics.get('frozen_threshold_logit', metrics.get('calibration', {}).get('threshold_logit'))
        assert threshold == report['threshold_logit']
        native[directory.name] = metrics
        test_receipts[directory.name] = dict(receipt, frozen_threshold_logit=threshold,
                                             exporter_sha256=metrics.get('exporter_sha256'))
        checks.append({'check': 'checkpoint_bytes_manifest_and_aggregate_counts', 'name': directory.name, 'pass': True})
        print('verified checkpoint / aggregate receipt:', directory.name, flush=True)

    mapping = {'rf_s_s17': 'S', 'rf_s_s7': 'S_s7', 'rf_s_s27': 'S_s27',
               'rf_sjg_s17': 'SJ_gentle', 'rf_sjg_s7': 'SJ_gentle_s7', 'rf_sjg_s27': 'SJ_gentle_s27'}
    for name, arm in mapping.items():
        p = runs / arm / 'out'
        r, c, hist = read(p / 'RUN_VERIFIED.json'), read(p / 'config.json'), read(p / 'history.json')
        assert (p / 'DONE').exists() and r['status'] == 'COMPLETED_VERIFIED'
        assert len(hist) == r['epochs_completed'] == c['epochs'] == 10
        feasible = [h for h in hist if h['validation']['overall']['auc'] >= c['min_auc']
                    and h['validation']['overall']['fnr'] <= c['max_fnr']]
        chosen = min(feasible, key=lambda h: (h['validation']['axes']['gender_race']['F_FPR'],
                                            -h['validation']['overall']['auc'], h['epoch']))
        assert chosen['epoch'] == r['selected_epoch']
        assert chosen['validation'] == r['validation']
        assert chosen['calibration']['threshold_logit'] == test_receipts[name]['frozen_threshold_logit']
        assert r['selected_checkpoint_sha256'] == test_receipts[name]['checkpoint_sha256']
        digest = hashlib.sha256(json.dumps(c, sort_keys=True, allow_nan=False).encode()).hexdigest()
        assert digest == r['config_digest']
        val[name] = r
        configs[name] = {'config': c, 'file_sha256': sha(p / 'config.json'), 'config_digest': digest,
                         'cluster_run_directory': str(p), 'run_receipt': read(p / 'RUN.json')}
        for h in hist:
            epochs.append({'name': name, 'seed': c['seed'], 'epoch': h['epoch'], 'feasible': h['feasible'],
                           'selected_this_epoch': h['selected_this_epoch'], 'calibration': h['calibration'],
                           'validation': h['validation'], 'epoch_seconds': h['epoch_seconds'],
                           'losses_step_mean': h['losses_step_mean']})
        checks.append({'check': 'fixed_selection_config_digest_threshold_and_checkpoint_binding', 'name': name, 'pass': True})

    for filename in ('METRICS_LONG_TEST.csv', 'GROUP_SUPPORT_TEST.csv'):
        shutil.copyfile(tests / 'metrics' / filename, out / filename)
    with (out / 'METRICS_LONG_TEST.csv').open() as f:
        table = list(csv.DictReader(f))
    assert len(table) == 80
    with (out / 'GROUP_SUPPORT_TEST.csv').open() as f:
        support = list(csv.DictReader(f))
    assert len(support) == 540
    models = {'rf_s_s17': ('RF-S (ours)', 's17'), 'rf_s_s7': ('RF-S (ours)', 's7'),
              'rf_s_s27': ('RF-S (ours)', 's27'), 'rf_sjg_s17': ('RF-SJg (ours)', 's17'),
              'rf_sjg_s7': ('RF-SJg (ours)', 's7'), 'rf_sjg_s27': ('RF-SJg (ours)', 's27'),
              'ding_gda': ('Ding corrected GDA', 's17'), 'daw': ('DAW-FDD', 's17'),
              'cesfl_p': ('CES-FL P', 's17'), 'l_a': ('L_A (plain-head BCE)', 's17')}
    for name, (model, seed) in models.items():
        d = read(tests / 'out' / name / 'metrics.json')['calibrated']['overall']
        for line in (r for r in table if r['model'] == model and r['seed'] == seed):
            for key in ('auc', 'fpr', 'fnr'):
                assert math.isclose(float(line['overall_' + key + '_pct']), 100 * d[key], abs_tol=1e-10)
    checks.append({'check': '80_csv_rows_match_native_overall_metrics_540_support_rows', 'pass': True})

    # Capture an independently completed duplicate, but never replace a frozen TEST checkpoint.
    duplicate_path = root / 'ff2_sjgentle_confirmation_20261108_r1/SJ_gentle_s7/out'
    dup = read(duplicate_path / 'RUN_VERIFIED.json')
    assert dup['validation'] == val['rf_sjg_s7']['validation']
    duplicates = {'primary_rule': 'Use the checkpoint paths in the existing TEST freeze for both VAL and TEST.',
                  'additional_seed7_run': str(duplicate_path),
                  'additional_checkpoint_sha256': dup['selected_checkpoint_sha256'],
                  'frozen_checkpoint_sha256': val['rf_sjg_s7']['selected_checkpoint_sha256'],
                  'selected_validation_metrics_identical': True, 'extra_training_not_an_independent_seed': True}
    ledger_path = root / 'ff2_baseline_suite_s17_20261103T110324Z/additional72_gpu_hours.json'
    ledger = read(ledger_path)
    charged = sum(float(x.get('charged_gpu_hours', 0)) for x in ledger['jobs'].values())
    reserved = sum(float(x.get('reserved_gpu_hours', 0)) for x in ledger['jobs'].values()
                   if x.get('state') in ('RESERVED', 'RUNNING'))
    budget = {'source_path': str(ledger_path), 'source_sha256': sha(ledger_path),
              'cap_gpu_hours': ledger['cap_gpu_hours'], 'charged_gpu_hours': charged,
              'active_reserved_gpu_hours': reserved, 'unreserved_balance_gpu_hours': ledger['cap_gpu_hours']-charged-reserved,
              'test_inference_and_sanity_runtime_reconciliation': 'PENDING: receipts lack full start/end durations',
              'balance_is_not_a_complete_study_compute_total': True}
    # Only numeric aggregate probe data: no source images, IDs, or row scores.
    probe = read(root / 'ff2_source_diag_20261106/recrop_summary.json')
    write(out / 'PROCESSING_PROBE.json', probe)
    write(out / 'VALIDATION_SELECTED.json', val)
    write(out / 'EPOCH_HISTORY_AGGREGATE.json', epochs)
    write(out / 'CONFIGURATIONS.json', configs)
    write(out / 'TEST_RECEIPTS.json', test_receipts)
    write(out / 'TEST_NATIVE_AGGREGATE.json', native)
    write(out / 'DUPLICATE_RUN_AUDIT.json', duplicates)
    write(out / 'BUDGET_SNAPSHOT.json', budget)
    sources = {str(p.relative_to(repo)): sha(p) for p in (repo/'CES_FL/src/ces_fl_realfpr.py',
               repo/'CES_FL/src/ces_fl_head_ablation.py')}
    provenance = {'observed_utc': datetime.now(timezone.utc).isoformat(),
                  'reviewed_upstream_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                  'test_frozen_list_commit': '8d06aa09de29f3a6a7f666e0ab41c0cc3e7048ab',
                  'test_manifest_sha256': manifest_sha, 'source_files': sources, 'checks': checks,
                  'private_scores_read': False, 'images_read': False, 'models_executed': False,
                  'temporal_freeze_audit': 'Node clock skew prevents independent proof of push-before-first-TEST-read.',
                  'exporter_sanity': 'Four family representatives; small DAW/GDA drift accepted under revised VAL-only tolerance.',
                  'repeat_scope': 'Downstream seeds 7/17/27 share the same seed17 BCE50 encoder; SD is not a confidence interval.'}
    write(out / 'PROVENANCE.json', provenance)
    print(json.dumps({'checks_passed': len(checks), 'test_checkpoints': 10, 'selected_validation_runs': 6,
                      'test_summary_rows': len(table), 'test_group_rows': len(support), 'budget': budget}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    a = p.parse_args()
    collect(a.runs_root, a.out, a.repo)
