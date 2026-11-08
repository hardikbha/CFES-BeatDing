# FairForensics Table III layout for the one-time FF2 public TEST read (6 frozen models, pre-registered
# 8 Nov 2026). Aggregates only, from metrics/METRICS_LONG_TEST.csv; no new inference. Mirrors the VAL table builder
# (~/ces_fl_runs/ff2_fairness_tables_20261105/build_table3_style.py) but adds explicit n_seeds disclosure per row,
# since 4 of 6 rows are single-seed and 2 are 3-seed mean +/- std.
import sys
from pathlib import Path
import pandas as pd

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(Path(sys.argv[2]) / 'METRICS_LONG_TEST.csv', keep_default_na=False)
sub = m[m.axis == 'sex_age_race8_all'].copy().reset_index(drop=True)

GROUPS = [
    ('Fair-based', [("DAW-FDD WACV'24", 'DAW-FDD'), ("Ding et al. CVPR'26 (corrected GDA, our run)", 'Ding corrected GDA')]),
    ('Ours\n(CES-FL, head ablation)', [('CES-FL P: original-only BCE', 'CES-FL P'),
                                        ('L\\_A: plain-head BCE (matched control)', 'L_A (plain-head BCE)'),
                                        ('RF-S: real-only FPR equalisation', 'RF-S (ours)'),
                                        ('\\textbf{RF-SJg: RF-S + gentle JPEG aug}', 'RF-SJg (ours)')]),
]
COLS = [('F_FPR_pct', 'min', r'$F_{\mathrm{FPR}}\!\downarrow$'), ('F_MEO_max_pp', 'min', r'$F_{\mathrm{MEO}}\!\downarrow$'),
        ('F_DP_pct', 'min', r'$F_{\mathrm{DP}}\!\downarrow$'), ('F_OAE_pp', 'min', r'$F_{\mathrm{OAE}}\!\downarrow$'),
        ('overall_auc_pct', 'max', r'AUC$\uparrow$'), ('es_AUC_pct', 'max', r'es-AUC$\uparrow$'),
        ('FPR_gap_pp', 'min', r'FPR gap$\downarrow$'), ('TPR_gap_pp', 'min', r'TPR gap$\downarrow$'),
        ('worst_group_auc_pct', 'max', r'Worst-group AUC$\uparrow$')]

g = sub.groupby('model')
mean = g[[c for c, _, _ in COLS]].mean()
std = g[[c for c, _, _ in COLS]].std()
nseeds = g.size()

allkeys = [k for _, rows in GROUPS for _, k in rows]
ranked = {}
for col, d, _ in COLS:
    order = mean.loc[allkeys, col].astype(float).sort_values(ascending=(d == 'min'))
    ranked[col] = list(order.index)

lines = []
for gname, rows in GROUPS:
    for i, (label, key) in enumerate(rows):
        cells = []
        for col, d, _ in COLS:
            v = float(mean.loc[key, col])
            sd = std.loc[key, col]
            t = f'{v:.2f}' + (f'\\scriptsize$\\pm${sd:.2f}' if pd.notna(sd) else '')
            rank = ranked[col].index(key)
            if rank == 0: t = f'\\textbf{{{t}}}'
            elif rank == 1: t = f'\\underline{{{t}}}'
            cells.append(t)
        n = int(nseeds[key])
        lead = f'\\multirow{{{len(rows)}}}{{*}}{{\\shortstack{{{gname}}}}}' if i == 0 else ''
        lines.append(f'{lead} & {label} & ' + ' & '.join(cells) + f' & {n} \\\\')
    lines.append('\\midrule')
body = '\n'.join(lines[:-1])
header = ' & '.join(h for _, _, h in COLS)

doc = r'''\documentclass[11pt]{article}
\usepackage[margin=1cm,landscape]{geometry}
\usepackage{booktabs,multirow,graphicx,amsmath}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\begin{document}
\begin{table}[h]\centering
\caption{Detection and fairness performance (\%) on \textbf{FairForge-2 public TEST} (sealed split, read exactly once,
8 Nov 2026, pre-registered). Fairness is computed over the 8 sex $\times$ age ($<$30/30+) $\times$ race
(White/Non-White) intersectional subgroups, as in FairForensics (arXiv 2608.01661) Table III, plus four extra
diagnostic columns: es-AUC, the plain FPR/TPR max$-$min gap in percentage points, and the AUC of the single
worst-performing group. Every checkpoint's own VAL-selected threshold is frozen; none is recalibrated on TEST.
\textbf{Bold}/\underline{underline} = best/second-best mean per column. The rightmost column gives the number of
training seeds behind each row: 4 of 6 rows are a single seed (17), disclosed as such; RF-S and RF-SJg are
mean $\pm$ std over seeds 17/7/27.}
\resizebox{\linewidth}{!}{%
\begin{tabular}{ll ccccc ccc c c}
\toprule
& & \multicolumn{5}{c}{FairForensics Table III metrics} & \multicolumn{4}{c}{Extra diagnostics} & \\
\cmidrule(lr){3-7}\cmidrule(lr){8-11}
Model Type & Detector & ''' + header + r''' & Seeds \\
\midrule
''' + body + r'''
\bottomrule
\end{tabular}}
\end{table}
\end{document}
'''
(OUT / 'FF2_TEST_TABLE3_STYLE.tex').write_text(doc)
print('ok')
