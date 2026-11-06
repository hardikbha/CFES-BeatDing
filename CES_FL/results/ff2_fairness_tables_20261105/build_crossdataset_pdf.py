# FairForensics-style cross-dataset tables (Table III layout). Aggregates only; reads existing seed-mean metrics.
import sys
from pathlib import Path
import pandas as pd

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
D = Path('/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DEMO_AXES/results/fairforensics_ids')
s = pd.read_csv(D / 'ids8_seedmean.csv', keep_default_na=False)
s = s[(s.threshold == 'A1') & (s.labels == 'main') & (s.rule == 'min_support')]
SETS = [('ffpp_test', 'FF++ (in-domain)'), ('celebdf', 'Celeb-DF'), ('dfdc', 'DFDC'), ('dfd', 'DFD'), ('ff2_test', 'FairForge-2 TEST')]
MET = [('F_FPR', 'min'), ('F_MEO', 'min'), ('F_DP', 'min'), ('F_OAE', 'min'), ('AUC', 'max')]
GROUPS = [('Fair-based', 'Xception', [('Ding et al.\\ CVPR\'26 Ori (our repro.)', 'Ding Ori repro (SFD)'),
                                      ('Ding et al.\\ CVPR\'26 SFD+GDA (our repro.)', 'Ding repro (SFD+GDA)')]),
          ('Ours (CFES)', 'CLIP ViT-L/14', [('Detection-only control', 'control (det-only)'), ('CFES A (rev4)', 'CFES A (rev4)'),
                                            ('arm C', 'arm C'), ('arm E w1', 'arm E w1'), ('arm E w3', 'arm E w3'),
                                            ('arm E w10', 'arm E w10'), ('\\textbf{CFES-D (final)}', 'CFES-D')]),
          ('Ours (CFES)', 'ResNet-50', [('CFES-ResNet', 'CFES-ResNet')])]
key = lambda m, st: s[(s.model == m) & (s.set == st)]
best = {}
for st, _ in SETS:
    for mk, d in MET:
        vals = {m: float(key(m, st).iloc[0][mk]) for _, _, rs in GROUPS for _, m in rs if len(key(m, st))}
        best[(st, mk)] = (min if d == 'min' else max)(vals.values())
lines = []
for gname, bb, rows in GROUPS:
    for i, (label, m) in enumerate(rows):
        seeds = int(key(m, 'ffpp_test').iloc[0].n_seeds)
        cells = []
        for st, _ in SETS:
            r = key(m, st)
            for mk, _ in MET:
                v = float(r.iloc[0][mk]); t = f'{v:.2f}'
                cells.append(f'\\textbf{{{t}}}' if f'{v:.2f}' == f'{best[(st, mk)]:.2f}' else t)
        lead = (f'\\multirow{{{len(rows)}}}{{*}}{{\\shortstack[l]{{{gname}\\\\{bb}}}}}' if i == 0 else '')
        lines.append(f'{lead} & {label} & {seeds} & ' + ' & '.join(cells) + ' \\\\')
    lines.append('\\midrule')
t1 = '\n'.join(lines[:-1])
head = ' & '.join(f'\\multicolumn{{5}}{{c|}}{{{n}}}' if k < len(SETS) - 1 else f'\\multicolumn{{5}}{{c}}{{{n}}}' for k, (_, n) in enumerate(SETS))
sub = ' & '.join(['$F_{\\mathrm{FPR}}\\!\\downarrow$ & $F_{\\mathrm{MEO}}\\!\\downarrow$ & $F_{\\mathrm{DP}}\\!\\downarrow$ & $F_{\\mathrm{OAE}}\\!\\downarrow$ & AUC$\\uparrow$'] * len(SETS))

ROWS2 = [('Ori (BCE)', 'done (VAL)'), ('DAG-FDD', 'done (VAL)'), ('DAW-FDD', 'done (VAL)'), ('PG-FDD', 'not trained'),
         ('F3Net (FAD)', 'trained; verification failed'), ('CORE', 'trained; verification failed'),
         ('Ding et al.\\ (corrected GDA, our run)', 'done (VAL)'), ('CES-FL P', 'done (VAL)'), ('CES-FL A', 'done (VAL)'),
         ('CES-FL B', 'done (VAL)'), ('L\\_A (matched control)', 'done (VAL)'), ('\\textbf{L\\_H-PS (ours)}', 'awaiting probe review')]
t2 = '\n'.join(f'{n} & {st} & \\multicolumn{{20}}{{c}}{{\\textcolor{{gray}}{{-- not evaluated --}}}} \\\\' for n, st in ROWS2)
sets2 = SETS[:4]
head2 = ' & '.join(f'\\multicolumn{{5}}{{c|}}{{{n.replace(" (in-domain)", "")}}}' if k < 3 else f'\\multicolumn{{5}}{{c}}{{{n}}}' for k, (_, n) in enumerate(sets2))
sub2 = ' & '.join(['$F_{\\mathrm{FPR}}$ & $F_{\\mathrm{MEO}}$ & $F_{\\mathrm{DP}}$ & $F_{\\mathrm{OAE}}$ & AUC'] * 4)

doc = r'''\documentclass[9pt]{extarticle}
\usepackage[margin=1.2cm,landscape,a4paper]{geometry}
\usepackage{booktabs,multirow,graphicx,xcolor,amsmath}
\usepackage[T1]{fontenc}\usepackage{lmodern}
\title{Cross-dataset fairness tables in the FairForensics (Table III) format: current status}
\author{CES-FL project, 6 Nov 2026. Aggregates of existing saved scores; no new inference.}
\date{}
\begin{document}\maketitle
\noindent\fcolorbox{orange}{orange!8}{\parbox{0.98\linewidth}{\textbf{Format.} As FairForensics (arXiv 2608.01661) Table III:
per test set, $F_{\mathrm{FPR}}$, $F_{\mathrm{MEO}}$, $F_{\mathrm{DP}}$, $F_{\mathrm{OAE}}$ over 8 intersectional demographic
subgroups (sex $\times$ age 0--29/30+ $\times$ White/Non-White, labelled with the official FairFace classifier, as in their
protocol) and AUC, all in \%. \textbf{Table 1 (filled):} train on FF++ (Ding protocol), test on FF++ and four unseen datasets
including FairForge-2. \textbf{Table 2 (empty):} the exact FairForensics analogue, train on FairForge-2 TRAIN and test on
external datasets; no FF2-trained model has been evaluated there yet. Threshold: each model's FF++-validation rule (A1).
Video sets use video-majority labels; a group enters FPR terms only with $\geq$3 real videos (persons on FF2). Values are seed
means (seed count shown). These are descriptive; no significance claims. Bold = best per column among rows shown.}}

\begin{table}[h]\centering
\caption{Performance comparison under cross-dataset evaluation, trained on FF++ (Ding et al.\ protocol). Detection and
fairness (\%) on FF++ (in-domain), Celeb-DF, DFDC, DFD and FairForge-2 public TEST (87{,}564 images; every FF2 generator is
unseen for these models). Lower fairness scores are better; higher AUC is better.}
\resizebox{\linewidth}{!}{\begin{tabular}{ll c|ccccc|ccccc|ccccc|ccccc|ccccc}\toprule
Type & Detector & Seeds & ''' + head + r''' \\
 & & & ''' + sub + r''' \\\midrule
''' + t1 + r'''
\bottomrule\end{tabular}}\end{table}

\noindent\textbf{Reading Table 1.} On the FairForge-2 column the arm E variants reach 77--79 AUC (one seed each)
and CFES-D 70.99 (3 seeds), against 36--38 for the Ding reproductions. FairForge-2 is hard for every FF++-trained
detector, and Ding's method transfers worst. These rows are the older CFES (CLIP) lineage; the current CES-FL L\_A/L\_H
(Xception, plain head) have Ding-format FF++ results but have not yet been rescored on these 8 groups (CPU only, from saved scores).

\begin{table}[h]\centering
\caption{The exact FairForensics analogue: train on FairForge-2 public TRAIN (218{,}266 images), test on unseen datasets.
\textbf{Status: not evaluated.} Checkpoints marked ``done (VAL)'' exist and are verified on FF2 validation; none has been run on
the external test sets. That one-time evaluation (inference only, frozen checkpoints and thresholds) needs approval.}
\resizebox{\linewidth}{!}{\begin{tabular}{ll|ccccc|ccccc|ccccc|ccccc}\toprule
Detector (trained on FF2) & FF2 checkpoint & ''' + head2 + r''' \\
 & & ''' + sub2 + r''' \\\midrule
''' + t2 + r'''
\bottomrule\end{tabular}}\end{table}
\end{document}
'''
(OUT / 'CROSSDATASET_TABLES_DRAFT.tex').write_text(doc)
print('ok')
