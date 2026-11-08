"""Build public tables/figures from the committed aggregate snapshot. No model/data access."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'evidence'
OUT = ROOT / 'generated'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def load(name):
    return json.loads((E / name).read_text())


def summary(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def fmt(values, tex=True):
    mean, sd = summary(values)
    if sd is None:
        return f'{mean:.2f}'
    return (f'${mean:.2f} \\pm {sd:.2f}$' if tex else f'{mean:.2f} ± {sd:.2f}')


with (E / 'METRICS_LONG_TEST.csv').open() as f:
    rows = list(csv.DictReader(f))
with (E / 'GROUP_SUPPORT_TEST.csv').open() as f:
    groups = list(csv.DictReader(f))
MODELS = ['Ding corrected GDA', 'DAW-FDD', 'CES-FL P', 'L_A (plain-head BCE)', 'RF-S (ours)', 'RF-SJg (ours)']
LABELS = ['Corrected GDA$^\\dagger$', 'DAW-FDD$^\\dagger$', 'CES-FL P$^\\dagger$', '$L_A$ (BCE)', 'RF-S', 'RF-SJg']
FIELDS = ['overall_auc_pct', 'overall_fpr_pct', 'overall_fnr_pct', 'F_FPR_pct', 'worst_group_auc_pct']
data = {(m, a): [r for r in rows if r['model'] == m and r['axis'] == a]
        for m in MODELS for a in {r['axis'] for r in rows}}
assert len(rows) == 80 and len(groups) == 540
for m in MODELS:
    d = data[m, 'gender_race']
    assert len(d) == (3 if 'ours' in m else 1)

table = [r'\begin{tabular}{lcrrrrr}', r'\toprule',
         r'Method & $n$ & AUC$\uparrow$ & FPR$\downarrow$ & FNR$\downarrow$ & $100F_{\rm FPR}\downarrow$ & WG-AUC$\uparrow$ \\', r'\midrule']
markdown = ['# Frozen FairForge-2 TEST summary', '',
            'All rates/AUC are multiplied by 100. F_FPR is a sum of absolute group deviations, not a single error rate.',
            '± denotes sample SD over downstream seeds 7/17/27 sharing one seed17 pretrained encoder; it is not a CI.', '',
            '| Method | Seeds | AUC | Achieved FPR | FNR | Gender×race 100F_FPR | Worst-group AUC |',
            '|---|---:|---:|---:|---:|---:|---:|']
for m, label in zip(MODELS, LABELS):
    d = data[m, 'gender_race']
    table.append(label + ' & ' + str(len(d)) + ' & ' + ' & '.join(fmt([float(r[k]) for r in d]) for k in FIELDS) + r' \\')
    markdown.append('| ' + m + ' | ' + str(len(d)) + ' | ' + ' | '.join(fmt([float(r[k]) for r in d], False) for k in FIELDS) + ' |')
table += [r'\bottomrule', r'\end{tabular}']
(OUT / 'test_primary.tex').write_text('\n'.join(table) + '\n')

axes = ['gender', 'race', 'gender_race', 'age_bin', 'skin_bin_known', 'age_x_skin16', 'sex_age_race8_min20']
axis_labels = ['Gender', 'Race', 'Gender $\\times$ race', 'Age (estimated)', 'Skin (estimated)', 'Age $\\times$ skin', 'Sex $\\times$ age $\\times$ race']
axis_table = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
              r'Axis & GDA$^\dagger$ & DAW$^\dagger$ & P$^\dagger$ & $L_A$ & RF-S & RF-SJg \\', r'\midrule']
markdown += ['', '## Supported axes: 100F_FPR (lower is better)', '',
             '| Axis | Corrected GDA | DAW-FDD | CES-FL P | L_A | RF-S | RF-SJg |', '|---|---:|---:|---:|---:|---:|---:|']
for a, label in zip(axes, axis_labels):
    vals = [[float(r['F_FPR_pct']) for r in data[m, a]] for m in MODELS]
    axis_table.append(label + ' & ' + ' & '.join(fmt(v) for v in vals) + r' \\')
    markdown.append('| ' + label.replace('$', '').replace('\\times', '×') + ' | ' + ' | '.join(fmt(v, False) for v in vals) + ' |')
axis_table += [r'\bottomrule', r'\end{tabular}']
(OUT / 'test_axes.tex').write_text('\n'.join(axis_table) + '\n')

val = load('VALIDATION_SELECTED.json')
validation = [r'\begin{tabular}{lcrrrr}', r'\toprule',
              r'Method & Seed & Epoch & AUC$\uparrow$ & FNR$\downarrow$ & $100F_{\rm FPR}\downarrow$ \\', r'\midrule']
for name in sorted(val):
    r = val[name]; v = r['validation']; method = 'RF-SJg' if 'sjg' in name else 'RF-S'
    validation.append(f"{method} & {name.split('_')[-1][1:]} & {r['selected_epoch']} & {100*v['overall']['auc']:.4f} & "
                      f"{100*v['overall']['fnr']:.4f} & {100*v['axes']['gender_race']['F_FPR']:.4f}" + r' \\')
validation += [r'\bottomrule', r'\end{tabular}']
(OUT / 'validation_selected.tex').write_text('\n'.join(validation) + '\n')

support = [r'\begin{tabular}{lrr}', r'\toprule', r'Gender $\times$ race & Real & Fake \\', r'\midrule']
for r in groups:
    if r['model'] == 'Ding corrected GDA' and r['axis'] == 'gender_race':
        support.append(r['group'] + ' & ' + r['n_real'] + ' & ' + r['n_fake'] + r' \\')
support += [r'\bottomrule', r'\end{tabular}']
(OUT / 'test_support.tex').write_text('\n'.join(support) + '\n')

paired = {}
for field in FIELDS:
    diffs = []
    for seed in ('s17', 's7', 's27'):
        a = next(r for r in data['RF-S (ours)', 'gender_race'] if r['seed'] == seed)
        b = next(r for r in data['RF-SJg (ours)', 'gender_race'] if r['seed'] == seed)
        diffs.append(float(b[field]) - float(a[field]))
    paired[field] = {'SJg_minus_S_per_seed_pp': dict(zip(('17','7','27'), diffs)),
                     'mean_difference_pp': statistics.mean(diffs), 'sd_difference_pp': statistics.stdev(diffs)}
(OUT / 'PAIRED_SEED_DESCRIPTIVE.json').write_text(json.dumps(paired, indent=2) + '\n')

# Plot numeric aggregates only. Error bars are seed SD, not image/contributor CIs.
fig, ax = plt.subplots(figsize=(6.7, 3.1))
colors = ['#666666', '#e6a200', '#9c6bbe', '#b84848', '#3476ad', '#14836c']
for m, label, color in zip(MODELS, ['Corrected GDA', 'DAW-FDD', 'CES-FL P', 'L_A', 'RF-S', 'RF-SJg'], colors):
    d = data[m, 'gender_race']; x = [float(r['overall_auc_pct']) for r in d]; y = [float(r['F_FPR_pct']) for r in d]
    mx,sx = summary(x); my,sy = summary(y)
    ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt='o', color=color, capsize=3)
    ax.annotate(label, (mx,my), xytext=(5, 5 if label not in ('DAW-FDD','CES-FL P') else -13), textcoords='offset points', fontsize=8)
ax.set(xlabel='TEST AUC (%) — higher is better', ylabel='Gender × race 100 F_FPR — lower is better', xlim=(92.7,99.55), ylim=(0,75))
ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/'test_tradeoff.pdf'); plt.close(fig)

probe = load('PROCESSING_PROBE.json')
keys = ['A0_stored','A3_fairface_geometry','A4_fairface_geometry_jpeg75_420']
fig, ax = plt.subplots(figsize=(6.7,2.8)); positions = np.arange(3)
for k, color, shift, label in [('reals','#3476ad',-.18,'Real false positives (n=242)'),('fakes','#b84848',.18,'Fake detections (n=360)')]:
    values = [100*probe[k][x]['flag_fake_rate'] for x in keys]
    ax.bar(positions+shift, values, .36, label=label, color=color)
    for pos,v in zip(positions+shift,values): ax.text(pos,v+1,f'{v:.1f}',ha='center',fontsize=8)
ax.set_xticks(positions, ['Stored crop','Matched geometry\nwithout JPEG','Matched geometry\n+ q75 / 4:2:0'])
ax.set(ylabel='Classified fake (%)',ylim=(0,125));ax.legend(loc='upper center',fontsize=8,ncol=2)
ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(OUT/'processing_probe.pdf');plt.close(fig)

fig, ax = plt.subplots(figsize=(6.7,3.3))
mat=np.array([[statistics.mean(float(r['F_FPR_pct']) for r in data[m,a]) for a in axes] for m in MODELS])
im=ax.imshow(mat,cmap='YlOrRd',aspect='auto',vmin=0,vmax=float(mat.max()))
for i in range(len(MODELS)):
    for j in range(len(axes)): ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',fontsize=8,color='black')
ax.set_yticks(range(6),['Corrected GDA','DAW-FDD','CES-FL P','L_A','RF-S','RF-SJg'])
ax.set_xticks(range(7),['Gender','Race','G×R','Age*','Skin*','Age×skin*','S×A×R*'])
fig.colorbar(im,ax=ax,label='100 F_FPR (sum; lower is better)',pad=.015)
fig.tight_layout();fig.savefig(OUT/'test_fairness_axes.pdf');plt.close(fig)

(ROOT / 'RESULTS.md').write_text('\n'.join(markdown)+'\n')
print(json.dumps({'status':'PASS','source':'committed aggregate snapshot only','test_summary_rows':len(rows),
                  'group_rows':len(groups),'tables':4,'figures':3,'paired_difference':paired['F_FPR_pct']},indent=2))
