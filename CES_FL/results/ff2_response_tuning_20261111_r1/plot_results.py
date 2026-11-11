"""Plot stored aggregate SELECT trajectories; no inference or private rows."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
with (ROOT/'ALL_EPOCHS_SELECT.csv').open(newline='') as f:history=list(csv.DictReader(f))
with (ROOT/'SELECTED_VALIDATION_RESULTS.csv').open(newline='') as f:selected={r['candidate']:r for r in csv.DictReader(f)}
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
for name,label,color in [('fair_only','Fairness-only control','#333333'),('smooth_005','Smooth cap0.05','#27854A'),('smooth_010','Smooth cap0.10','#255E91'),('smooth_020','Smooth cap0.20','#C45624')]:
    rows=[r for r in history if r['candidate']==name];s=selected[name]
    for ax,key in zip(axes,('AUC_pct','joint_100F_FPR')):
        ax.plot([int(r['epoch']) for r in rows],[float(r[key]) for r in rows],label=label,color=color,marker='o',markersize=3)
        ax.scatter(int(s['selected_epoch']),float(s[key]),color=color,marker='*',s=150,zorder=4,edgecolors='white',linewidths=.7)
axes[0].set(ylabel='SELECT AUC (%)',title='Full validation trajectories')
axes[1].set(ylabel='Joint100F_FPR (lower is better)',title='Primary disparity across epochs')
for ax in axes:
    ax.axvline(2.5,linestyle='--',color='#999999',linewidth=1)
    ax.set_xlabel('Epoch');ax.set_xticks([1,2,3,4,5,8,10]);ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
fig.suptitle('Seed17 SELECT only; stars mark AUC-selected checkpoints; primary uncertainty is inconclusive',fontsize=10)
fig.savefig(ROOT/'VALIDATION_TRAJECTORIES.pdf',bbox_inches='tight')
fig.savefig(ROOT/'VALIDATION_TRAJECTORIES.png',dpi=180,bbox_inches='tight')
