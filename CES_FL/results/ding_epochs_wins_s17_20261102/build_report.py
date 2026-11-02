"""Describe existing epoch budgets and count displayed-paper comparisons; no ML execution."""
import argparse
from collections import Counter
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

AXES = ('gender', 'race', 'gender_race')
METRICS = ('F_FPR', 'F_DP', 'es_AUC')
DATASETS = {'FF++': 'ffpp', 'DFDC': 'dfdc', 'Celeb-DF': 'celebdf', 'DFD': 'dfd'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_training(path, expected_epoch=None, replay_path=None):
    path = Path(path)
    config = json.loads((path / 'config.json').read_text())
    receipt_path = Path(replay_path) if replay_path else path / 'RUN_VERIFIED.json'
    verified = json.loads(receipt_path.read_text())
    history = [json.loads(line) for line in (path / 'history.jsonl').read_text().splitlines() if line.strip()]
    assert verified['status'] == 'PASS' and (path / 'DONE').exists()
    assert [h['epoch'] for h in history] == list(range(1, config['epochs'] + 1))
    selected = max(history, key=lambda h: h['validation']['overall']['auc'])['epoch']
    assert selected == verified.get('selected_epoch', selected)
    if replay_path:
        assert verified['absolute_auc_difference'] == 0 and verified['threshold_unchanged']
        assert verified['recorded_validation_auc'] == max(h['validation']['overall']['auc'] for h in history)
    if expected_epoch is not None:
        assert selected == expected_epoch
    return {'path': str(path), 'configured_epochs': config['epochs'], 'completed_epochs': len(history),
            'selected_epoch': selected, 'checkpoint_sha256_from_receipt': verified['checkpoint_sha256'],
            'config_sha256': digest(path / 'config.json'),
            'history_sha256': digest(path / 'history.jsonl'),
            'receipt_path': str(receipt_path), 'receipt_sha256': digest(receipt_path)}


def outcome(ours, paper, direction):
    if ours is None or paper is None:
        return 'undefined'
    difference = (ours - paper) * (1 if direction == 'higher' else -1)
    return 'tie' if abs(difference) <= 1e-12 else ('win' if difference > 0 else 'loss')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not any((args.out / n).exists() for n in ['REPORT.md', 'comparison.json', 'METRIC_COMPARISONS.csv']), 'Refuse overwrite'
    s = json.loads(args.snapshot.read_text())
    manifests = json.loads((args.snapshot.parent / 'ARTIFACT_SHA256.json').read_text())
    assert digest(args.snapshot) == manifests['source_snapshot.json']
    pretraining = {
        'Xception': inspect_training('/home/sachin.chaudhary/ces_fl_pipeline/audits/encoder_controls_20261030T202634/full_experiment_runs/P_pretrain'),
        'ResNet-50': inspect_training('/home/sachin.chaudhary/ces_fl_pipeline/experiments/benchmark_s17_20261031T112401/full_experiment_runs/P_resnet50'),
    }
    published = {}
    rows = s['paper']['tables']['1']
    published[('Xception', 'ffpp')] = sum([r['values_pct'][:3] for r in rows if r['method'] == 'Ding et al. (published)'], []) + [97.71]
    for table, backbone in [('2', 'Xception'), ('3', 'ResNet-50')]:
        for row in s['paper']['tables'][table]:
            if row['method'] == 'Ding et al. (published)':
                published[(backbone, DATASETS[row['block']])] = row['values_pct']
    arms, comparisons = [], []
    for e in s['experiments']:
        replay = (Path('/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_tests_s17_20261101T040407Z/validation_audit') /
                  e['id'] / 'replay.json') if e['study'] == 'upgrade_20261101' else None
        training = inspect_training(Path(e['checkpoint_path']).parent, e['selected_epoch'], replay)
        assert training['config_sha256'] == e['config_sha256']
        assert training['checkpoint_sha256_from_receipt'] == e['checkpoint_sha256']
        has_pretrain = e['config'].get('encoder_checkpoint') is not None
        if has_pretrain:
            assert e['config']['encoder_checkpoint_sha256'] == pretraining[e['backbone']]['checkpoint_sha256_from_receipt']
        a = {'id': e['id'], 'label': e['label'], 'backbone': e['backbone'], 'seed': 17,
             'training': training, 'pretraining': pretraining[e['backbone']] if has_pretrain else None,
             'reports': {}}
        for report in ['calibrated', 'fixed_probability_0_5']:
            records = []
            for ds, d in e['datasets'].items():
                m = d[report]
                assert m['units'] == 'fractions'
                values = [(axis, key, None if m['axes'][axis]['ding_style_fractions'] is None else
                           100 * m['axes'][axis]['ding_style_fractions'][key], 'lower' if key != 'es_AUC' else 'higher')
                          for axis in AXES for key in METRICS]
                values.append(('overall', 'AUC', 100 * m['overall']['auc'], 'higher'))
                assert len(values) == 10
                for (axis, key, ours, direction), paper in zip(values, published[(e['backbone'], ds)]):
                    r = {'arm': e['id'], 'backbone': e['backbone'], 'seed': 17, 'dataset': ds, 'report': report,
                         'axis': axis, 'metric': key, 'direction': direction, 'ours_pct': ours, 'ding_published_pct': paper,
                         'ours_minus_ding_pp': None if ours is None else ours - paper,
                         'outcome': outcome(ours, paper, direction)}
                    records.append(r)
            assert len(records) == 40
            counts = dict(Counter(r['outcome'] for r in records))
            per_dataset = {ds: dict(Counter(r['outcome'] for r in records if r['dataset'] == ds)) for ds in DATASETS.values()}
            per_metric = {key: dict(Counter(r['outcome'] for r in records if r['metric'] == key)) for key in (*METRICS, 'AUC')}
            # Table1 prints the same AUC three times; report that raw count separately.
            ffpp_wins = per_dataset['ffpp'].get('win', 0)
            ffpp_auc_win = next(r['outcome'] == 'win' for r in records if r['dataset'] == 'ffpp' and r['metric'] == 'AUC')
            a['reports'][report] = {'counts': counts, 'per_dataset': per_dataset, 'per_metric': per_metric,
                                    'fairness_wins_of_36': sum(r['outcome'] == 'win' for r in records if r['metric'] != 'AUC'),
                                    'auc_wins_of_4': per_metric['AUC'].get('win', 0),
                                    'table1_raw_wins_of_12': ffpp_wins + 2 * ffpp_auc_win if e['backbone'] == 'Xception' else None}
            comparisons.extend(records)
        arms.append(a)
    assert outcome(1, 2, 'lower') == outcome(2, 1, 'higher') == 'win'
    assert outcome(1, 1, 'higher') == 'tie' and outcome(None, 1, 'lower') == 'undefined'
    assert len(comparisons) == 880
    b = next(a for a in arms if a['id'] == 'B_detection_frozen')
    c = next(a for a in arms if a['id'] == 'C_detection_invariance')
    assert b['reports']['calibrated']['counts']['win'] == c['reports']['calibrated']['counts']['win'] == 17
    report = {'created_management_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'snapshot_path': str(args.snapshot), 'snapshot_sha256': digest(args.snapshot),
              'method': '40 unique cells: four datasets × (three attributes × three fairness metrics + overall AUC once)',
              'tie_tolerance_in_percentage_points': 1e-12,
              'main_operating_point': 'Previously stored validation-only calibrated threshold; never selected using test',
              'first_epoch_to_surpass_ding': 'UNAVAILABLE: no TEST comparison evaluated for every epoch',
              'training_launched': False, 'inference_launched': False, 'pretraining': pretraining, 'arms': arms}
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (args.out / 'METRIC_COMPARISONS.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(comparisons[0]), lineterminator='\n')
        w.writeheader(); w.writerows(comparisons)
    md = ['# CES-FL: training epochs and numerical comparisons with Ding\n',
          'Prepared 2 November 2026. These are the existing seed17 TEST results in [Tables 1–3](../ding_format_tables_s17_20261102/README.md). No training, inference, calibration or test-based selection was performed for this report.\n',
          '**Epochs:** the reported detection-pretrained models completed 10 original-only pretraining epochs and 10 downstream epochs per arm. Pretraining is shared by the arms, not rerun for each one. ImageNet initialization predates these stages. A uses no deepfake pretraining.\n',
          '**We cannot identify the first epoch at which a model exceeded Ding.** Only validation-selected frozen checkpoints were tested; a selected epoch is not a measured crossover epoch. Do not choose a different checkpoint by comparing epochs on TEST.\n',
          '## Verified pretraining\n',
          '| Backbone | Completed epochs | Validation-selected epoch | Checkpoint SHA256 |',
          '|---|---:|---:|---|']
    for bb, p in pretraining.items():
        md.append(f"| {bb} | {p['completed_epochs']} | {p['selected_epoch']} | `{p['checkpoint_sha256_from_receipt']}` |")
    md += ['\n## Completed downstream arms and numerical wins\n',
           'Counts compare unrounded stored measurements with Ding\'s displayed, rounded published numbers for the same backbone. They are descriptive point comparisons, not statistical wins or proof of protocol equivalence. Main rows use the previously stored validation-calibrated thresholds.\n',
           '| Arm | Backbone | Completed downstream epochs | Validation-selected epoch | Wins / 40 | Ties / 40 | Fairness wins / 36 | AUC wins / 4 | F_FPR wins / 12 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in arms:
        q = a['reports']['calibrated'];t = a['training']
        md.append(f"| {a['id']} | {a['backbone']} | {t['completed_epochs']} | {t['selected_epoch']} | {q['counts'].get('win',0)} | {q['counts'].get('tie',0)} | {q['fairness_wins_of_36']} | {q['auc_wins_of_4']} | {q['per_metric']['F_FPR'].get('win',0)} |")
    md += ['\n## Wins by dataset\n',
           '| Arm | Backbone | FF++ / 10 | DFDC / 10 | Celeb-DF / 10 | DFD / 10 |',
           '|---|---|---:|---:|---:|---:|']
    for a in arms:
        q = a['reports']['calibrated']['per_dataset']
        md.append('| ' + ' | '.join([a['id'], a['backbone']] + [str(q[ds].get('win',0)) for ds in DATASETS.values()]) + ' |')
    md += ['\n## Counting and interpretation\n',
           'For Xception, Table1 supplies FF++ and Table2 supplies the other three datasets. For ResNet-50, Table3 supplies all four datasets. Each dataset contributes nine fairness cells plus one overall AUC: 40 unique comparisons. Table1 repeats overall AUC in the Gender, Race and Intersection blocks; counting all 12 printed cells would double-count AUC. The JSON records that raw Table1 count separately.\n',
           'A strict win means lower F_FPR/F_DP or higher es-AUC/AUC. Absolute differences up to 1e-12 percentage points are numerical ties. Paper rounding and experimental uncertainty are not resolved by this tolerance. No checkpoint, threshold or preferred arm was changed after counting.\n',
           '**Fairness remains mixed.** Xception B and C both have 17/40 total numerical wins. C has 16/36 fairness wins versus B\'s 15/36, but only one of four AUC wins versus B\'s two. Both exceed Ding on just 1/12 F_FPR cells. RD has 15/40 total wins and 0/12 F_FPR wins. These counts do not support an overall fairness-superiority claim.\n',
           'B and upgrade R0 repeat the same measured scores and are not independent seeds. C is invariance-only; RN/RD are six-source diagnostics. The current source `58773bb` 50+50 study is validation-only and excluded. Counts here are not the separate historical BD0 preregistered endpoint.\n',
           'Our operating points, training policies, support filtering and architecture differ from the published report; the same table layout alone does not establish a matched-protocol comparison. [Full protocol caveats](../ding_format_tables_s17_20261102/README.md).\n',
           '[Exact per-metric comparisons](METRIC_COMPARISONS.csv) include both saved calibrated and fixed0.5 reports without selecting between them. [Epoch/config/history hashes and counts](comparison.json). Training histories, DONE and native verification receipts were read; no large checkpoint was reloaded or rehashed.\n',
           '## Actual command and checks\n',
           '```bash\n/home/sachin.chaudhary/.conda/envs/aiface/bin/python CES_FL/results/ding_epochs_wins_s17_20261102/build_report.py --snapshot CES_FL/results/ding_format_tables_s17_20261102/source_snapshot.json --out CES_FL/results/ding_epochs_wins_s17_20261102\n```\n',
           'Actual successful output: `PASS: 13 training histories; 11 arms; 880 metric comparisons (two stored operating points); 40 unique cells per arm/report; no training/inference.` Histories contain epochs 1–10, all completion receipts pass, and selected epochs match earliest highest validation AUC. Existing output files are never overwritten.\n']
    (args.out / 'REPORT.md').write_text('\n'.join(md))
    print('PASS: 13 training histories; 11 arms; 880 metric comparisons (two stored operating points); 40 unique cells per arm/report; no training/inference.')


if __name__ == '__main__':
    main()
