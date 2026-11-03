"""Aggregate validation-rate heatmaps; no model/images/test inputs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(reports, out):
    out=Path(out);out.mkdir(exist_ok=True)
    for axis in ['gender','race','gender_race','age_bin','skin_bin']:
        groups=reports['D']['axes'][axis]['eligible_groups']
        assert all(v['axes'][axis]['eligible_groups']==groups for v in reports.values())
        fig,axs=plt.subplots(2,1,figsize=(max(7,len(groups)*1.6),max(5,len(reports)*1.0+3)),layout='constrained')
        for ax,key,denom in zip(axs,['fpr','fnr'],['n_real','n_fake']):
            values=np.asarray([[r['axes'][axis]['groups'][g][key]*100 for g in groups] for r in reports.values()])
            im=ax.imshow(values,vmin=0,vmax=50,cmap='Reds',aspect='auto')
            ax.set_xticks(range(len(groups)),groups,rotation=20,ha='right');ax.set_yticks(range(len(reports)),list(reports))
            for i,r in enumerate(reports.values()):
                for j,g in enumerate(groups):
                    n=r['axes'][axis]['groups'][g][denom]
                    ax.text(j,i,f'{values[i,j]:.2f}%\nn={n}',ha='center',va='center',fontsize=8,color='white' if values[i,j]>30 else 'black')
            ax.set_title(axis+' '+key.upper()+' at each frozen selected-VAL threshold')
            fig.colorbar(im,ax=ax,label=key.upper()+' (%)',fraction=.025,pad=.01)
        fig.suptitle('CES-FL validation only; correlated frames, inherited group labels',fontsize=11)
        fig.savefig(out/(axis+'_rates.png'),dpi=150);plt.close(fig)
