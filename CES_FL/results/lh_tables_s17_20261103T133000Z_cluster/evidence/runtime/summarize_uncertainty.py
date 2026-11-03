"""Publish existing aggregate bootstrap outputs; performs no ML or inference."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ('ffpp', 'dfdc', 'celebdf', 'dfd')
VIEWS = ('fixed_probability_0.5', 'native_selected_VAL_FPR10')


def read(path):
    return json.loads(Path(path).read_text())


def interval_text(interval):
    if interval['estimate'] is None:
        return 'undefined'
    value = f"{100 * interval['estimate']:+.3f}"
    if interval['ci95'] is not None:
        lo, hi = interval['ci95']
        value += f" [{100 * lo:+.3f}, {100 * hi:+.3f}]"
    return value + f"; n={interval['valid_replicates']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    done = read(ROOT / 'ANALYSIS_DONE.json')
    assert done['status'] == 'PASS'
    files = {f'{a}_{d}' for a in ('L_A', 'L_H') for d in DATASETS}
    files.update(f'H_minus_A_{d}' for d in DATASETS)
    assert {p.stem for p in (ROOT / 'analysis').glob('*.json')} == files
    rows, support, loaded = [], [], {}
    hashes = {}
    for name in sorted(files):
        path = ROOT / 'analysis' / (name + '.json')
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        record = loaded[name] = read(path)
        paired = name.startswith('H_minus_A_')
        dataset = name.removeprefix('H_minus_A_') if paired else record['dataset']
        for view in VIEWS:
            uncertainty = record[view] if paired else record[view]['uncertainty']
            assert uncertainty['replicates'] == 1000 and uncertainty['seed'] == 17
            support.append(dict(analysis=name, dataset=dataset, view=view,
                                components=uncertainty['components'],
                                largest_component_rows=uncertainty['largest_component_rows']))
            for metric, value in uncertainty['intervals'].items():
                ci = value['ci95']
                rows.append(dict(analysis=name, dataset=dataset, view=view,
                                 metric=metric, estimate_fraction=value['estimate'],
                                 lower95_fraction=None if ci is None else ci[0],
                                 upper95_fraction=None if ci is None else ci[1],
                                 usable_draws=value['valid_replicates'],
                                 unusable_draws=value['invalid_or_unavailable_replicates'],
                                 meets_900_draw_requirement=value['valid_replicates'] >= 900,
                                 training_seed_count=1))
    for name, data in [('BOOTSTRAP_INTERVALS.csv', rows), ('BOOTSTRAP_SUPPORT.csv', support)]:
        with (out / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(data)
    lines = ['# Locked benchmark uncertainty', '',
             datetime.datetime.now(datetime.timezone.utc).isoformat() + ' management UTC.', '',
             'All eight native analyses and four paired H-minus-A analyses completed. '
             'Each view uses 1,000 connected-recorded-contributor draws, seed 17, '
             'fixed original group eligibility and unchanged stored thresholds. '
             'The interval is a percentile interval conditional on these fixed trained models. '
             'It measures dataset-sampling uncertainty, not training-seed variability.', '',
             'Numbers below are 100 times fraction values: AUC/FPR/FNR differences are '
             'percentage points; F_FPR is a sum of absolute disparities and is not a probability. '
             'The n value is usable draws out of 1,000. Cells with fewer than 900 usable draws '
             'do not meet the retained uncertainty requirement, even when an interval is displayed. '
             'An interval crossing zero does not prove equivalence. These tables introduce no new '
             'checkpoint, threshold or training-selection rule.', '']
    collections = [
        ('Detection differences', ['overall/auc', 'overall/fpr', 'overall/fnr']),
        ('Primary demographic differences', ['gender/F_FPR', 'race/F_FPR', 'gender_race/F_FPR']),
        ('Auxiliary demographic differences', ['age_bin/F_FPR', 'skin_bin/F_FPR']),
    ]
    for view in VIEWS:
        lines += [f'## {view}', '']
        for title, metrics in collections:
            lines += [f'### {title}: H minus A', '',
                      '| Dataset | ' + ' | '.join(metrics) + ' |',
                      '|---|' + '---|' * len(metrics)]
            for dataset in DATASETS:
                intervals = loaded['H_minus_A_' + dataset][view]['intervals']
                lines.append('| ' + dataset + ' | ' + ' | '.join(interval_text(intervals[k]) for k in metrics) + ' |')
            lines.append('')
        lines += ['### Resampling support', '', '| Dataset | Components | Largest component (rows) |', '|---|---:|---:|']
        for dataset in DATASETS:
            u = loaded['H_minus_A_' + dataset][view]
            lines.append(f"| {dataset} | {u['components']} | {u['largest_component_rows']} |")
        lines.append('')
    low = [r for r in rows if r['estimate_fraction'] is not None and r['usable_draws'] < 900]
    lines += [f'{len(low)} defined model/view/metric cells have fewer than 900 usable draws. '
              'See BOOTSTRAP_INTERVALS.csv for every cell, including undefined values and all supported subgroup intervals.', '',
              'Recorded contributors are not verified biological identities. FF++ retains the explicitly '
              'declared historical fit:TEST contributor overlap; DFDC retains 1,476 repeated native '
              'manifest rows. Connected resampling accounts for recorded image/source/identity links '
              'but cannot repair incomplete provenance. Age and skin-tone labels retain their original proxy-label limitations.', '',
              'A small FF++ AUC difference or a lower FPR at fixed 0.5 is not sufficient to claim a win over Ding. '
              'Compare missed fakes and both operating-point views; published and historical Ding rows have '
              'different protocols. No new training decision is made from these TEST outcomes.', '']
    (out / 'UNCERTAINTY.md').write_text('\n'.join(lines))
    receipt = dict(status='PASS', management_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   analysis_files=hashes, source_commit=done['source_commit'],
                   single_model_analyses=8, paired_dataset_analyses=4,
                   bootstrap_draws=1000, required_usable_draws=900,
                   interval_rows=len(rows), below_900_defined_cells=len(low),
                   no_model_or_threshold_selection=True, no_additional_inference=True)
    (out / 'ANALYSIS_VERIFIED.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('UNCERTAINTY_RENDER_PASS', len(files), 'analyses', len(rows), 'interval rows', len(low), 'low-support defined cells')


if __name__ == '__main__':
    main()
