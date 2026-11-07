# Renders the current FairForge-2 paper tables (VAL-only draft) to LaTeX. Aggregates only; no inference.
import json, sys
from pathlib import Path
import pandas as pd

OUT = Path(sys.argv[1]); RUNS = Path(sys.argv[2])
ST = Path('/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/release_ff2_public/stats')
DIAG = Path('/home/sachin.chaudhary/ces_fl_runs/ff2_fairness_tables_20261105/diag_source_fpr.json')

ROWS = [('Ori (BCE)', 'BCE'), ('DAG-FDD WACV\'24', 'DAG-FDD'), ('DAW-FDD WACV\'24', 'DAW-FDD'), ('PG-FDD CVPR\'24', None),
        ('F3Net (FAD branch)', None), ('CORE', None), ('Ding et al.\\ CVPR\'26 (corrected GDA, our run)', 'Ding corrected GDA'),
        ('CES-FL P: original-only BCE', 'CES-FL P'), ('CES-FL A: weighted BCE', 'CES-FL A'), ('CES-FL B: A + soft error disparity', 'CES-FL B'),
        ('L\\_A: plain-head BCE (matched control)', 'L_A (plain-head BCE)'),
        ('\\textbf{L\\_H-PS (ours): hierarchical-real + probe}', 'L_H-PS (ours)'),
        ('\\textbf{CES-FL full (ours, final)}', None)]
NOTE = {'PG-FDD CVPR\'24': 'not run on FF2', 'F3Net (FAD branch)': 'trained; verification failed', 'CORE': 'trained; verification failed',
        '\\textbf{CES-FL full (ours, final)}': 'not run on FF2',
        '\\textbf{L\\_H-PS (ours): hierarchical-real + probe}': 'awaiting human review of the probe'}

m = pd.read_csv(RUNS / 'METRICS_LONG_9RUNS.csv', keep_default_na=False)
m = m[m.threshold_view == 'VAL10']
g = pd.read_csv(RUNS / 'GROUP_SUPPORT_9RUNS.csv', keep_default_na=False)
g = g[g.threshold_view == 'VAL10']
dg = json.load(open(DIAG))


def val(model, axis, col):
    return float(m[(m.model == model) & (m.axis == axis)].iloc[0][col])


def within_source_ffpr(model, src, races):
    n, fp = dg[model][f'{src}|ALL']; ov = fp / n
    return 100 * sum(abs(dg[model][f'{src}|{r}'][1] / dg[model][f'{src}|{r}'][0] - ov) for r in races)


def table(cols, getters, dirs):
    # cols: header cells; getters: list of fn(model)->float; dirs: 'min'/'max'
    filled = [k for _, k in ROWS if k]
    best = []
    for fn, d in zip(getters, dirs):
        vals = {k: fn(k) for k in filled}
        best.append(min(vals, key=vals.get) if d == 'min' else max(vals, key=vals.get))
    lines = []
    for name, k in ROWS:
        if name.startswith('CES-FL P') or name.startswith('L\\_A'):
            lines.append('\\midrule')
        if k is None:
            lines.append(f'{name} & \\multicolumn{{{len(getters)}}}{{c}}{{\\textcolor{{gray}}{{-- pending ({NOTE[name]}) --}}}} \\\\')
        else:
            cells = [f'\\textbf{{{fn(k):.2f}}}' if f'{fn(k):.2f}' == f'{fn(best[i]):.2f}' else f'{fn(k):.2f}' for i, fn in enumerate(getters)]
            lines.append(name + ' & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines)


# Table 2: Ding format
AX = [('gender', 'g'), ('race', 'r'), ('gender_race', 'i')]
getters, dirs = [], []
for ax, _ in AX:
    for col, d in [('F_FPR_pct', 'min'), ('F_DP_pct', 'min'), ('es_AUC_pct', 'max')]:
        getters.append(lambda k, ax=ax, col=col: val(k, ax, col)); dirs.append(d)
getters.append(lambda k: val(k, 'gender', 'overall_auc_pct')); dirs.append('max')
t2 = table(None, getters, dirs)

# Table 3: extra axes
g3, d3 = [], []
for ax in ['age_bin', 'skin_bin_known']:
    for col, d in [('F_FPR_pct', 'min'), ('es_AUC_pct', 'max'), ('worst_group_auc_pct', 'max')]:
        g3.append(lambda k, ax=ax, col=col: val(k, ax, col)); d3.append(d)
g3.append(lambda k: within_source_ffpr(k, 'img', ['Asian', 'Black', 'White', 'Others'])); d3.append('min')
g3.append(lambda k: within_source_ffpr(k, 'ffhq', ['Asian', 'Black', 'White'])); d3.append('min')
g3.append(lambda k: val(k, 'gender', 'overall_fnr_pct')); d3.append('min')
t3 = table(None, g3, d3)

# Table 4: per-group FPR (gender x race)
GR = ['Female-Asian', 'Female-Black', 'Female-White', 'Female-Others', 'Male-Asian', 'Male-Black', 'Male-White', 'Male-Others']
gr = g[g.axis == 'gender_race']
g4 = [lambda k, grp=grp: float(gr[(gr.model == k) & (gr.group == grp)].iloc[0].fpr_pct) for grp in GR]
t4 = table(None, g4, ['min'] * len(GR))
nreal = {grp: int(gr[(gr.model == 'BCE') & (gr.group == grp)].iloc[0].n_real) for grp in GR}

# Table 1: composition
c = pd.read_csv(ST / 'groups_split_realfake_cellvalid.csv').merge(pd.read_csv(ST / 'groups_source_realfake_cellvalid.csv'), on='group')
c = c.set_index('group').loc[GR]
f = lambda x: f'{int(x):,}'
t1 = '\n'.join(f"{grp} & {f(r.TRAIN_real)} / {f(r.TRAIN_fake)} & {f(r.SELECT_real)} / {f(r.SELECT_fake)} & {f(r.CAL_real)} / {f(r.CAL_fake)} & "
               f"{f(r.TEST_real)} / {f(r.TEST_fake)} & {f(r.FFHQ_real)} & {f(r.FairFace_real)} \\\\" for grp, r in c.iterrows())
tot = c.sum()
t1 += (f"\n\\midrule\nAll & {f(tot.TRAIN_real)} / {f(tot.TRAIN_fake)} & {f(tot.SELECT_real)} / {f(tot.SELECT_fake)} & {f(tot.CAL_real)} / "
       f"{f(tot.CAL_fake)} & {f(tot.TEST_real)} / {f(tot.TEST_fake)} & {f(tot.FFHQ_real)} & {f(tot.FairFace_real)} \\\\")

doc = r'''\documentclass[10pt]{article}
\usepackage[margin=1.6cm,landscape]{geometry}
\usepackage{booktabs,multirow,graphicx,xcolor,amsmath}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\newcommand{\Fpr}{F_{\mathrm{FPR}}}\newcommand{\Fdp}{F_{\mathrm{DP}}}
\title{FairForge-2 tables: current draft}
\author{CES-FL project, updated 7 Nov 2026 (L\_A and L\_H-PS added) from verified run aggregates}
\date{}
\begin{document}\maketitle
\noindent\fcolorbox{orange}{orange!8}{\parbox{0.97\linewidth}{\textbf{Status of the numbers.} All results are on FairForge-2
\textbf{SELECT (validation), not TEST}: seed 17, one run per method, threshold fixed on SELECT reals for 10\% FPR (achieved
9.97\%). TEST stays sealed until the final model, checkpoint and threshold are frozen; the paper version will report TEST
(87{,}564 images, plus the 16{,}953-image unseen-generator subset) as mean $\pm$ std over 3 seeds. Grey rows are not yet
available. Bold = best among the filled rows. Age and skin bins are machine estimates (InsightFace age, ITA skin tone),
not human labels. CES-FL P/A/B are controls of the CES-FL recipe; L\_A (6 Nov) is the matched BCE control for L\_H-PS (plain linear head, encoder from FF2 BCE50, 10 epochs, selected epoch 2); L\_H-PS (7 Nov) is our fairness method on the same recipe (selected epoch 10). Pre-declared rule: L\_H-PS is fairer than L\_A only if gender$\times$race $F_{\mathrm{FPR}}$ drops by $\geq$2 points with AUC loss $\leq$0.5 and FNR $\leq$2$\times$: not met (37.37$\to$36.23); the full counterfactual CES-FL objective has not run on FF2.}}

\begin{table}[h]\centering
\caption{FairForge-2 public split: real / fake images with valid group labels, per gender$\times$race group and split, and
real images per source dataset. ``Others'' (Indian, Latino/Hispanic, Middle Eastern) exists only in FairFace.}
\small\begin{tabular}{l|cccc|cc}\toprule
Group & TRAIN & SELECT & CAL & TEST & FFHQ reals & FairFace reals \\\midrule
''' + t1 + r'''
\bottomrule\end{tabular}\end{table}

\begin{table}[h]\centering
\caption{Main table in the format of Ding \etal{} (values $\times$100). g = gender (2 groups), r = race (4), i = gender$\times$race
(8). $\Fpr=\sum_g |\mathrm{FPR}_g-\mathrm{FPR}|$, es-AUC $=\mathrm{AUC}/(1+\sum_g|\mathrm{AUC}_g-\mathrm{AUC}|)$. FairForge-2 SELECT, seed 17.}
\resizebox{\linewidth}{!}{\begin{tabular}{l|ccc|ccc|ccc|c}\toprule
& \multicolumn{9}{c|}{Fairness metrics (\%)} & Detection (\%) \\
Method (all Xception) & g $\Fpr\!\downarrow$ & g $\Fdp\!\downarrow$ & g es-AUC$\uparrow$ & r $\Fpr\!\downarrow$ & r $\Fdp\!\downarrow$ & r es-AUC$\uparrow$ & i $\Fpr\!\downarrow$ & i $\Fdp\!\downarrow$ & i es-AUC$\uparrow$ & AUC$\uparrow$ \\\midrule
''' + t2 + r'''
\bottomrule\end{tabular}}\end{table}

\begin{table}[h]\centering
\caption{Axes beyond Ding \etal{} (values $\times$100). Age: $<$30, 30--44, 45--59, 60+. Skin: 4 ITA bins (unknown excluded).
Race $\Fpr$ within source recomputes the race metric inside one source dataset, against that source's own FPR
(FairFace: 4 races, 2{,}196 reals; FFHQ: 3 races, 242 reals). FNR = overall miss rate.}
\resizebox{\linewidth}{!}{\begin{tabular}{l|ccc|ccc|cc|c}\toprule
& \multicolumn{3}{c|}{Age (4 bins)} & \multicolumn{3}{c|}{Skin tone (4 bins)} & \multicolumn{2}{c|}{Race $\Fpr\!\downarrow$ within source} & \\
Method (all Xception) & $\Fpr\!\downarrow$ & es-AUC$\uparrow$ & worst AUC$\uparrow$ & $\Fpr\!\downarrow$ & es-AUC$\uparrow$ & worst AUC$\uparrow$ & FairFace & FFHQ & FNR$\downarrow$ \\\midrule
''' + t3 + r'''
\bottomrule\end{tabular}}\end{table}

\begin{table}[h]\centering
\caption{False-positive rate (\%) on real images per gender$\times$race group (lower is better; the overall target is 10\%).
Real images per group in brackets. FairForge-2 SELECT, seed 17.}
\resizebox{\linewidth}{!}{\begin{tabular}{l|cccc|cccc}\toprule
Method (all Xception) & ''' + ' & '.join(f'{x.replace("-", " ")} ({nreal[x]})' for x in GR) + r''' \\\midrule
''' + t4 + r'''
\bottomrule\end{tabular}}\end{table}

\noindent\textbf{Reading the tables.} Ding-style GDA has the lowest $\Fpr$ on gender, race, gender$\times$race and age, but the
lowest AUC and a 3--5$\times$ higher miss rate. No CES-FL control is best on any $\Fpr$ column. Most of the pooled race and
age disparity comes from the source dataset: FFHQ reals are flagged as fake far more often than FairFace reals
(Table~3, within-source columns; README T9 in the results branch).
\end{document}
'''
(OUT / 'FF2_TABLES_DRAFT.tex').write_text(doc.replace('\\etal{}', 'et al.'))
print('ok')
