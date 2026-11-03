#!/usr/bin/env python3
"""Audit existing aggregate reports only; no training, inference or threshold search.

Usage: python analyze_published.py --out /new/private/or/local/audit --plot
The optional plot requires numpy/matplotlib. Core analysis uses the standard library.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f'Published report inconsistency: {a} != {b}')


def check_report(report):
    overall = report['overall']
    if overall['n'] != overall['n_real'] + overall['n_fake']:
        raise ValueError('Overall class counts disagree')
    for axis, payload in report['axes'].items():
        groups = payload['groups']
        for field in ('n', 'n_real', 'n_fake'):
            if sum(g[field] for g in groups.values()) != overall[field]:
                raise ValueError(f'Incomplete partition: {axis}/{field}')
        for g in groups.values():
            if g['n'] != g['n_real'] + g['n_fake']:
                raise ValueError('Group class counts disagree')
            for rate, count in (('fpr', 'n_real'), ('fnr', 'n_fake')):
                if g[count]:
                    if g[rate] is None or not 0 <= g[rate] <= 1:
                        raise ValueError('Invalid supported group rate')
                    # Rates must describe an integer number of mistakes.
                    close(g[rate] * g[count], round(g[rate] * g[count]))
                elif g[rate] is not None:
                    raise ValueError('Unsupported class has a defined error rate')
        fp = sum(g['n_real'] * (g['fpr'] or 0.) for g in groups.values())
        fn = sum(g['n_fake'] * (g['fnr'] or 0.) for g in groups.values())
        close(fp, overall['n_real'] * overall['fpr'])
        close(fn, overall['n_fake'] * overall['fnr'])
        eligible = payload['eligible_groups']
        if any(g not in groups or min(groups[g]['n_real'], groups[g]['n_fake']) < payload['min_support_per_class'] for g in eligible):
            raise ValueError('Invalid eligible-group support')
        if len(eligible) > 1:
            close(payload['F_FPR'], sum(abs(groups[g]['fpr'] - overall['fpr']) for g in eligible))
            rates = [groups[g]['positive_rate'] for g in eligible]
            close(payload['F_DP'], max(rates) - min(rates))
            close(payload['es_AUC'], overall['auc'] / (1 + sum(abs(groups[g]['auc'] - overall['auc']) for g in eligible)))


def summarize(data):
    if data['format'] != 'CESFL_CELEBDF_PUBLISHED_AUDIT_v1' or data['new_model_inference']:
        raise ValueError('Unexpected audit input')
    reports = data['reports']
    for model in reports:
        for dataset, views in reports[model].items():
            for report in views.values():
                check_report(report)
            close(views['fixed_probability_0_5']['overall']['auc'], views['calibrated']['overall']['auc'])
    h = reports['L_H']['celebdf']['fixed_probability_0_5']
    a = reports['L_A']['celebdf']['fixed_probability_0_5']
    overall = h['overall']
    if (overall['n'], overall['n_real'], overall['n_fake']) != (28458, 5713, 22745):
        raise ValueError('Frozen Celeb-DF membership changed')
    for axis in h['axes']:
        for name, group in h['axes'][axis]['groups'].items():
            for field in ('n', 'n_real', 'n_fake'):
                if group[field] != a['axes'][axis]['groups'][name][field]:
                    raise ValueError('A/H demographic partitions disagree')
    joint = h['axes']['gender_race']
    contributions = []
    for name in joint['eligible_groups']:
        g = joint['groups'][name]
        amount = abs(g['fpr'] - overall['fpr'])
        contributions.append({'group': name, 'n_real': g['n_real'], 'n_fake': g['n_fake'],
                              'absolute_fpr_deviation': amount,
                              'share_of_reported_joint_F_FPR': amount / joint['F_FPR']})
    ideal_dp = {}
    for axis in ('gender', 'gender_race'):
        ax = h['axes'][axis]
        prevalences = [ax['groups'][g]['n_fake'] / ax['groups'][g]['n'] for g in ax['eligible_groups']]
        ideal_dp[axis] = max(prevalences) - min(prevalences)
    return {
        'status': 'PUBLISHED_AGGREGATES_RECONCILED',
        'source_commit': data['source_commit'], 'new_inference': False,
        'new_training': False, 'threshold_search': False,
        'counts': {k: overall[k] for k in ('n', 'n_real', 'n_fake')},
        'female_white_share_of_fakes': joint['groups']['Female-White']['n_fake'] / overall['n_fake'],
        'joint_F_FPR_contributions': contributions,
        'real_only_joint_groups': [g for g, v in joint['groups'].items() if v['n_real'] and not v['n_fake']],
        'hypothetical_perfect_prediction_DP_on_existing_eligible_groups': ideal_dp,
        'interpretation_of_perfect_DP': 'Algebraic base-rate diagnostic, not a trained model or replacement metric.',
        'overall_reports': {m: {v: reports[m]['celebdf'][v]['overall'] for v in reports[m]['celebdf']} for m in reports},
        'thresholds': {m: {v: reports[m]['celebdf'][v]['threshold_logit'] for v in reports[m]['celebdf']} for m in reports},
        'H_minus_A_auc_percentage_points': 100 * (overall['auc'] - a['overall']['auc']),
        'paired_intervals': data['paired_intervals'],
        'check_scope': 'Aggregate arithmetic and published membership counts only; private row alignment, original images and biological identity labels not verified here.',
    }


def plot(data, out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = data['reports']['L_H']['celebdf']['fixed_probability_0_5']
    groups = report['axes']['gender_race']['groups']
    names = ['Female-White', 'Male-Black', 'Female-Others', 'Male-White', 'Male-Others']
    values = np.array([[np.nan if groups[g][m] is None else 100 * groups[g][m]
                        for m in ('auc', 'fpr', 'fnr')] for g in names])
    colors = values.copy(); colors[:, 0] = 100 - colors[:, 0]
    cmap = plt.get_cmap('YlOrRd').copy(); cmap.set_bad('#e5e7eb')
    fig, ax = plt.subplots(figsize=(10.2, 5.7), layout='constrained')
    im = ax.imshow(np.ma.masked_invalid(colors), cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(3), ['AUC (%) ↑', 'False positives (%) ↓', 'Missed fakes (%) ↓'])
    ax.set_yticks(range(5), [f'{g}\n{groups[g]["n_real"]:,} real / {groups[g]["n_fake"]:,} fake' for g in names])
    for i in range(5):
        for j in range(3):
            label = 'Undefined\n(no fake samples)' if np.isnan(values[i, j]) else f'{values[i, j]:.2f}'
            ax.text(j, i, label, ha='center', va='center', fontsize=11,
                    color='white' if np.isfinite(colors[i, j]) and colors[i, j] > 65 else '#111827')
    fig.colorbar(im, ax=ax, label='Color severity: 100 − AUC, FPR or FNR')
    ax.set_title('Where L_H struggles on the fixed Celeb-DF benchmark\nOriginal selected checkpoint · probability threshold 0.5', pad=15)
    fig.supxlabel('Inherited manifest labels; not verified demographic ground truth. Point estimates only.\nGray cells are undefined. Real-only groups are excluded from the main joint fairness summary.\nPublished TEST snapshot 916999c · descriptive audit, no new model inference or tuning.', fontsize=9)
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'celebdf_group_diagnostics.{extension}', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=HERE / 'evidence/published_inputs.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Preserve earlier audit outputs; choose a new directory')
    data = json.loads(args.input.read_text())
    summary = summarize(data)
    summary['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    summary['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.mkdir(parents=True)
    (args.out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if args.plot:
        plot(data, args.out)
    print(json.dumps({k: summary[k] for k in ('status', 'source_commit', 'new_inference', 'new_training', 'threshold_search')}))


if __name__ == '__main__':
    main()
