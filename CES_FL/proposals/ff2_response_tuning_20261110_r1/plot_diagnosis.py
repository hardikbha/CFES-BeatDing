"""Plot existing SELECT curves and fixed schedules; no private/model inputs."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
with (ROOT/'PARENT_SELECT_CURVES.csv').open(newline='') as f:curves=list(csv.DictReader(f))
with (ROOT/'FIXED_SCHEDULES.csv').open(newline='') as f:schedules=list(csv.DictReader(f))
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
colors={'lh_ramp':'#C45624','lh_low':'#255E91'}
for method,label in [('lh_ramp','Existing ramp'),('lh_low','Existing low response')]:
    rows=[r for r in curves if r['method']==method];x=[int(r['epoch']) for r in rows]
    axes[0].plot(x,[float(r['VAL_AUC_pct']) for r in rows],marker='o',ms=3,label=label,color=colors[method])
    axes[1].plot(x,[float(r['response_classification_scalar_ratio']) for r in rows],marker='o',ms=3,label=label,color=colors[method])
for ax in axes[:2]:ax.axvline(2.5,linestyle='--',color='#777777',linewidth=1)
axes[0].set(title='Existing seed17 validation AUC',ylabel='SELECT AUC (%)')
axes[1].set(title='Weighted response / classification',ylabel='Mean loss-scalar ratio')
axes[1].axhline(1,color='#BBBBBB',linewidth=.8)
axes[2].plot(range(1,11),[0,0,1/3,2/3,1,1,1,1,1,1],color='#AAAAAA',linestyle=':',label='Existing ramp')
axes[2].plot(range(1,11),[0,0]+[1/3]*8,color='#777777',linestyle='--',label='Existing low response')
for name,label,color in [('fair_only','Fairness-only control','#333333'),('smooth_005','Smooth cap 0.05','#25854A'),('smooth_010','Smooth cap 0.10','#255E91'),('smooth_020','Smooth cap 0.20','#C45624')]:
    rows=[r for r in schedules if r['candidate']==name]
    axes[2].plot([int(r['epoch']) for r in rows],[float(r['multiplier']) for r in rows],label=label,color=color,linewidth=1.7)
axes[2].set(title='Fixed schedules being tested',ylabel='Response-loss multiplier')
for ax in axes:
    ax.set_xlabel('Epoch');ax.set_xticks([1,2,3,5,8,10]);ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,frameon=False)
fig.suptitle('Response activation at epoch3 coincides with validation degradation; causality remains unconfirmed',fontsize=11)
fig.savefig(ROOT/'SCHEDULE_DIAGNOSIS.pdf',bbox_inches='tight')
fig.savefig(ROOT/'SCHEDULE_DIAGNOSIS.png',dpi=180,bbox_inches='tight')
