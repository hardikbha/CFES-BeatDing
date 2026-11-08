# FairForensics Table III layout, single dataset (FairForge-2 VAL), 8 sex x age x race groups.
# Aggregates only, from already-computed METRICS_LONG_11RUNS.csv. No new inference.
import sys
from pathlib import Path
import pandas as pd

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
RUNS = Path(sys.argv[2])
m = pd.read_csv(RUNS / 'METRICS_LONG_11RUNS.csv', keep_default_na=False)
m = m[(m.threshold_view == 'VAL10') & (m.axis == 'sex_age_race8_all')].set_index('model')

PARAMS = {  # millions, matched-architecture parameter counts already verified in this project
    'BCE': 20.81, 'DAG-FDD': 20.81, 'DAW-FDD': 20.81, 'Ding corrected GDA': 20.81,
    'CES-FL P': 20.81, 'CES-FL A': 20.81, 'CES-FL B': 20.81,
    'L_A (plain-head BCE)': 20.81, 'L_H-PS (ours)': 20.81, 'RF-S (ours)': 20.81, 'RF-SJg (ours)': 20.81,
}
GROUPS = [
    ('CNN-based\n(Xception)', [('Ori (BCE)', 'BCE')]),
    ('Fair-based', [("DAG-FDD WACV'24", 'DAG-FDD'), ("DAW-FDD WACV'24", 'DAW-FDD'),
                     ("Ding et al. CVPR'26 (corrected GDA, our run)", 'Ding corrected GDA')]),
    ('Ours\n(CES-FL, head ablation)', [('CES-FL P: original-only BCE', 'CES-FL P'),
                                        ('CES-FL A: weighted BCE', 'CES-FL A'),
                                        ('CES-FL B: A + soft error disparity', 'CES-FL B'),
                                        ('L\\_A: plain-head BCE (matched control)', 'L_A (plain-head BCE)'),
                                        ('L\\_H-PS: hierarchical-real + probe', 'L_H-PS (ours)'),
                                        ('RF-S: real-only FPR equalisation', 'RF-S (ours)'),
                                        ('\\textbf{RF-SJg: RF-S + gentle JPEG aug}', 'RF-SJg (ours)')]),
]
COLS = [('F_FPR_pct', 'min', r'$F_{\mathrm{FPR}}\!\downarrow$'), ('F_MEO_max_pp', 'min', r'$F_{\mathrm{MEO}}\!\downarrow$'),
        ('F_DP_pct', 'min', r'$F_{\mathrm{DP}}\!\downarrow$'), ('F_OAE_pp', 'min', r'$F_{\mathrm{OAE}}\!\downarrow$'),
        ('overall_auc_pct', 'max', r'AUC$\uparrow$'), ('es_AUC_pct', 'max', r'es-AUC$\uparrow$'),
        ('FPR_gap_pp', 'min', r'FPR gap$\downarrow$'), ('TPR_gap_pp', 'min', r'TPR gap$\downarrow$'),
        ('worst_group_auc_pct', 'max', r'Worst-group AUC$\uparrow$')]

allkeys = [k for _, rows in GROUPS for _, k in rows]
ranked = {}
for col, d, _ in COLS:
    order = m.loc[allkeys, col].astype(float).sort_values(ascending=(d == 'min'))
    ranked[col] = list(order.index)

lines = []
for gname, rows in GROUPS:
    for i, (label, key) in enumerate(rows):
        cells = []
        for col, d, _ in COLS:
            v = float(m.loc[key, col]); t = f'{v:.2f}'
            rank = ranked[col].index(key)
            if rank == 0: t = f'\\textbf{{{t}}}'
            elif rank == 1: t = f'\\underline{{{t}}}'
            cells.append(t)
        lead = f'\\multirow{{{len(rows)}}}{{*}}{{\\shortstack{{{gname}}}}}' if i == 0 else ''
        lines.append(f'{lead} & {label} & ' + ' & '.join(cells) + f' & {PARAMS[key]:.2f} \\\\')
    lines.append('\\midrule')
body = '\n'.join(lines[:-1])
header = ' & '.join(h for _, _, h in COLS)

doc = r'''\documentclass[11pt]{article}
\usepackage[margin=1cm,landscape]{geometry}
\usepackage{booktabs,multirow,graphicx,amsmath}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\begin{document}
\begin{table}[h]\centering
\caption{Detection and fairness performance (\%) on \textbf{FairForge-2} (public SELECT/validation, seed 17), trained on
FairForge-2 public TRAIN. Fairness is computed over the 8 sex $\times$ age ($<$30/30+) $\times$ race (White/Non-White)
intersectional subgroups, as in FairForensics (arXiv 2608.01661) Table III, plus four extra diagnostic columns:
es-AUC (AUC discounted by its spread across groups), the plain FPR/TPR max$-$min gap in percentage points, and the
AUC of the single worst-performing group. Lower fairness scores and gaps indicate better demographic fairness; higher
AUC/es-AUC/worst-group AUC means better detection. \textbf{Bold} and \underline{underline} denote the best and
second-best results per column. All rows share one architecture (Xception, 20.81M parameters), so the comparison
isolates the training objective. These are validation, not TEST, scores; no result here has been evaluated on TEST.}
\resizebox{\linewidth}{!}{%
\begin{tabular}{ll ccccc ccc c c}
\toprule
& & \multicolumn{5}{c}{FairForensics Table III metrics} & \multicolumn{4}{c}{Extra diagnostics} & \\
\cmidrule(lr){3-7}\cmidrule(lr){8-11}
Model Type & Detector & ''' + header + r''' & Params (M) \\
\midrule
''' + body + r'''
\bottomrule
\end{tabular}}
\end{table}
\end{document}
'''
(OUT / 'FF2_TABLE3_STYLE.tex').write_text(doc)
print('ok')
