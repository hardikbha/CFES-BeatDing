# Builds the FF2 fairness table package (aggregates only) from existing result files. No inference, no TEST access.
# Inputs: the 5 Nov seven-run VAL aggregates (results branch), the frozen public-TEST benchmark CSVs, diag_source_fpr.json.
import json, sys
from pathlib import Path
import pandas as pd

OUT = Path(sys.argv[1])
RUNS = Path(sys.argv[2])  # folder holding METRICS_LONG_7RUNS.csv and GROUP_SUPPORT_7RUNS.csv
ST = Path('/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/release_ff2_public/stats')
DIAG = Path('/home/sachin.chaudhary/ces_fl_runs/ff2_fairness_tables_20261105/diag_source_fpr.json')
OUT.mkdir(parents=True, exist_ok=True)

ORDER = ['BCE', 'DAG-FDD', 'DAW-FDD', 'Ding corrected GDA', 'CES-FL P', 'CES-FL A', 'CES-FL B']
NAME = {'BCE': 'Xception BCE', 'DAG-FDD': 'DAG-FDD', 'DAW-FDD': 'DAW-FDD', 'Ding corrected GDA': 'Ding-style GDA (corrected)',
        'CES-FL P': 'CES-FL P (ours)', 'CES-FL A': 'CES-FL A (ours)', 'CES-FL B': 'CES-FL B (ours)'}
AXES = [('gender', 'Gender'), ('race', 'Race'), ('gender_race', 'Gender×Race'), ('age_bin', 'Age'), ('skin_bin_known', 'Skin')]


def md(df, floatfmt=2):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '|'.join('---:' if pd.api.types.is_numeric_dtype(df[c]) else '---' for c in cols) + '|']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f'{v:.{floatfmt}f}' if isinstance(v, float) else str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def bold_best(df, cols_dir):
    out = df.copy().astype(object)
    for c, d in cols_dir.items():
        v = df[c].astype(float)
        b = v.idxmin() if d == 'min' else v.idxmax()
        out[c] = [f'{x:.2f}' for x in v]
        out.loc[b, c] = f'**{v[b]:.2f}**'
    return out


sections = []
csvs = {}

# T1 composition
t1 = pd.read_csv(ST / 'groups_split_realfake_cellvalid.csv')
src = pd.read_csv(ST / 'groups_source_realfake_cellvalid.csv')
t1 = t1.merge(src, on='group')
csvs['T1_composition_by_group.csv'] = t1
sections.append(('T1. Composition: real/fake per gender×race group, split and source (public, cell-valid rows)',
                 md(t1), 'Others comes only from FairFace; Asian/Black/White mix FFHQ and FairFace. This matters for T9.'))

# T2 zero-shot panel on public TEST
p = pd.read_csv(ST / 'benchmarks/panel_public.csv')
p = p[~p.degenerate_public.astype(str).eq('True')].sort_values('AUC_public', ascending=False)
t2 = p[['model', 'type', 'AUC_public', 'iF_FPR_public', 'iF_MEO_public', 'iF_DP_public', 'iF_OAE_public', 'ies_AUC_public', 'FPR_public', 'TPR_public']]
t2.columns = ['Detector', 'Type', 'AUC', 'i-F_FPR', 'i-F_MEO', 'i-F_DP', 'i-F_OAE', 'i-es-AUC', 'FPR', 'TPR']
csvs['T2_zero_shot_panel_TEST.csv'] = t2
sections.append(('T2. Off-the-shelf detectors on FF2 public TEST (no FF2 training), non-degenerate detectors, sorted by AUC',
                 md(t2.reset_index(drop=True)), 'i- = gender×race intersection (8 groups). Released checkpoints at their own 0.5 rule.'))

# T3 Ding-format FF++ -> FF2 TEST
d = pd.read_csv(ST / 'benchmarks/ding_format_public.csv')
for sub, title in [('pub_all', 'all public TEST'), ('pub_unseen_models', 'TEST-only unseen models')]:
    t = d[(d.subset == sub) & (d.group == 'FF++-trained')].drop(columns=['subset', 'group', 'n', 'groups_with_both_classes'], errors='ignore')
    t = t.rename(columns={'model': 'Model', 'n_seeds': 'seeds'})
    csvs[f'T3_ding_format_ffpp_trained_{sub}_TEST.csv'] = t
    sections.append((f'T3{"a" if sub=="pub_all" else "b"}. Ding format, trained on FF++, tested on FF2 ({title})', md(t.reset_index(drop=True)),
                     'Values ×100. F_FPR, F_DP lower is better; es-AUC, AUC higher is better. Mean over the listed seeds.'))

# T4 family AUC
f = pd.read_csv(ST / 'benchmarks/family_auc_public.csv')
t4 = f[(f.subset == 'pub_all') & (f.group == 'FF++-trained')].drop(columns=['subset', 'group'])
csvs['T4_family_auc_ffpp_trained_TEST.csv'] = t4
sections.append(('T4. AUC per fake family, trained on FF++, FF2 public TEST', md(t4.reset_index(drop=True)), ''))

# T5 FF2-trained, VAL, all axes, all metrics
m = pd.read_csv(RUNS / 'METRICS_LONG_7RUNS.csv', keep_default_na=False)
m10 = m[m.threshold_view == 'VAL10']
rows = []
for mod in ORDER:
    s = m10[m10.model == mod].set_index('axis')
    r = {'Model': NAME[mod], 'AUC': float(s.loc['gender', 'overall_auc_pct']), 'FPR': float(s.loc['gender', 'overall_fpr_pct']),
         'FNR': float(s.loc['gender', 'overall_fnr_pct'])}
    for ax, lab in AXES:
        r[f'{lab} F_FPR'] = float(s.loc[ax, 'F_FPR_pct'])
    rows.append(r)
t5 = pd.DataFrame(rows).set_index('Model')
dirs = {c: ('max' if c == 'AUC' else 'min') for c in t5.columns}
csvs['T5_ff2_trained_FFPR_all_axes_VAL.csv'] = t5.reset_index()
sections.append(('T5. Trained on FF2: F_FPR on five axes, VAL (frozen VAL-real 10% FPR threshold)',
                 md(bold_best(t5, dirs).reset_index()), 'Values ×100; bold = best. Validation only, seed 17, one run each.'))

# T5b detailed per axis
for ax, lab in AXES:
    rows = []
    for mod in ORDER:
        s = m10[(m10.model == mod) & (m10.axis == ax)].iloc[0]
        rows.append({'Model': NAME[mod], 'groups': int(s.eligible_groups), 'F_FPR': float(s.F_FPR_pct), 'F_DP': float(s.F_DP_pct),
                     'es-AUC': float(s.es_AUC_pct), 'FPR gap': float(s.FPR_gap_pp), 'TPR gap': float(s.TPR_gap_pp),
                     'F_MEO': float(s.F_MEO_max_pp), 'F_OAE': float(s.F_OAE_pp), 'worst-group AUC': float(s.worst_group_auc_pct)})
    t = pd.DataFrame(rows).set_index('Model')
    dd = {c: ('max' if c in ('es-AUC', 'worst-group AUC') else 'min') for c in t.columns if c != 'groups'}
    csvs[f'T5_{ax}_detailed_VAL.csv'] = t.reset_index()
    sections.append((f'T5-{lab}. Detailed fairness metrics, {lab} axis, VAL', md(bold_best(t, dd).reset_index()), ''))

# T6 FairForensics-style 8 IDS
rows = []
for mod in ORDER:
    s = m10[(m10.model == mod) & (m10.axis == 'sex_age_race8_all')].iloc[0]
    rows.append({'Model': NAME[mod], 'AUC': float(s.overall_auc_pct), 'F_FPR': float(s.F_FPR_pct), 'F_MEO': float(s.F_MEO_max_pp),
                 'F_DP': float(s.F_DP_pct), 'F_OAE': float(s.F_OAE_pp)})
t6 = pd.DataFrame(rows).set_index('Model')
csvs['T6_fairforensics_ids8_VAL.csv'] = t6.reset_index()
sections.append(('T6. FairForensics-style 8 groups (sex × age<30/30+ × White/Non-White), VAL',
                 md(bold_best(t6, {'AUC': 'max', 'F_FPR': 'min', 'F_MEO': 'min', 'F_DP': 'min', 'F_OAE': 'min'}).reset_index()),
                 'Analogous grouping on FF2, not the FairForensics GenFace experiment. Male/Youth/White has only 18 VAL reals.'))

# T7/T8 per-group rates
g = pd.read_csv(RUNS / 'GROUP_SUPPORT_7RUNS.csv', keep_default_na=False)
g10 = g[g.threshold_view == 'VAL10']
for ax, lab in AXES + [('age_x_skin16', 'Age×Skin (16 cells)')]:
    sub = g10[g10.axis == ax]
    for col, cl in [('fpr_pct', 'FPR'), ('fnr_pct', 'FNR')]:
        t = sub.pivot(index='group', columns='model', values=col).astype(float)[ORDER].rename(columns=NAME)
        n = sub[sub.model == 'BCE'].set_index('group')[['n_real', 'n_fake']]
        t = n.join(t).reset_index()
        csvs[f'T8_{ax}_{cl}_per_group_VAL.csv'] = t
        if ax in ('gender_race', 'age_x_skin16', 'age_bin', 'skin_bin_known') or cl == 'FPR':
            sections.append((f'T{"7" if ax=="age_x_skin16" else "8"}. Per-group {cl} (%), {lab}, VAL', md(t), ''))

# T9 source confound diagnostic
dg = json.load(open(DIAG))
keys = ['img|ALL', 'ffhq|ALL'] + [f'{s}|{r}' for r in ['Asian', 'Black', 'White'] for s in ('ffhq', 'img')] + ['img|Others']
rows = []
for k in keys:
    s, c = k.split('|')
    r = {'Source': 'FairFace' if s == 'img' else 'FFHQ', 'Group': 'all' if c == 'ALL' else c, 'VAL reals': dg['BCE'][k][0]}
    for mod in ORDER:
        n, fp = dg[mod][k]
        r[NAME[mod]] = 100 * fp / n
    rows.append(r)
t9 = pd.DataFrame(rows)
csvs['T9_real_FPR_by_source_VAL.csv'] = t9
sections.append(('T9. Real-image false-positive rate (%) by SOURCE dataset, VAL — the main cause of the race/age gaps', md(t9),
                 'Each model uses its own frozen VAL-real 10% threshold. FFHQ = 242 of 2,438 VAL reals.'))

for name, df in csvs.items():
    df.to_csv(OUT / name, index=False, float_format='%.4f')
(OUT / 'TABLES.md').write_text('# FairForge-2 fairness tables (5 Nov 2026)\n\nGenerated by `build_tables.py` from existing aggregates. '
                               'No inference, no threshold tuning. TEST tables (T2–T4) use frozen public TEST scores of models not trained on FF2; '
                               'FF2-trained tables (T5–T9) are VALIDATION only.\n\n' +
                               '\n\n'.join(f'## {t}\n\n{body}\n\n{note}'.rstrip() for t, body, note in sections) + '\n')

# LaTeX for the two main tables
def tex(df, cap, label):
    cols = list(df.columns)
    s = ['\\begin{table}[t]\\centering\\small', f'\\caption{{{cap}}}\\label{{{label}}}',
         '\\begin{tabular}{l' + 'r' * (len(cols) - 1) + '}', '\\toprule', ' & '.join(cols).replace('_', '\\_').replace('×', '$\\times$') + ' \\\\', '\\midrule']
    for _, r in df.iterrows():
        s.append(' & '.join(str(v).replace('**', '').replace('_', '\\_') if not isinstance(v, float) else f'{v:.2f}' for v in r) + ' \\\\')
    s += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(s)

(OUT / 'tables.tex').write_text(tex(t5.reset_index(), 'FF2-trained detectors: F\\_FPR ($\\times$100, lower is better) on five axes, FF2 validation.', 'tab:ff2_ffpr_axes') + '\n\n' +
                                tex(t9, 'Real-image FPR (\\%) by source dataset on FF2 validation.', 'tab:ff2_source') + '\n')
print('wrote', len(csvs), 'csv;', len(sections), 'sections')
