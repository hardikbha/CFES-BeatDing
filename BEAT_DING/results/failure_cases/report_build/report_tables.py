"""Headline table + two spot checks for FAILURE_CASES.md (CPU, read-only on inputs)."""
import json, numpy as np, pandas as pd
R='/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents'
d=json.load(open(f'{R}/BEAT_DING/runs/BD0_s7/eval/bd_eval.json'))
cols=['gF_FPR','gF_DP','gesAUC','rF_FPR','rF_DP','resAUC','iF_FPR','iF_DP','iesAUC','AUC']
ding_files={'ffpp_test':'ff++','dfdc':'dfdc','celebdf':'celebdf','dfd':'dfd'}
print('thresholds',d['thresholds'])
for s in ['ffpp_test','dfdc','celebdf','dfd']:
    for op in ['matched','native','A1']:
        c=d['test'][s][op]['cols']
        print(s,op,'AUC %.2f TPR %.2f FPR %.2f iF_FPR %.2f wins %d/10'%(c['AUC'],c['TPR'],c['FPR'],c['iF_FPR'],d['test'][s][op]['n_wins']))
    pub=dict(zip(cols,d['published'][s])); print('  Ding published AUC %.2f iF_FPR %.2f'%(pub['AUC'],pub['iF_FPR']))
    z=np.load(f'{R}/DINGREPRO_LIN/results/audit_scores/full_s7_ep49_{ding_files[s]}.npz',allow_pickle=True)
    y=z['label']; p=z['logit']>=0
    print('  Ding repro@0.5 TPR %.2f FPR %.2f'%(100*p[y==1].mean(),100*p[y==0].mean()))
print('total wins',d['total_wins_vs_ding_published'])
# spot check: youtube_305 labels (FF++ label caveat)
L=pd.read_csv(f'{R}/DEMO_AXES/labels/labels_ffpp_test.csv')
m=L[L.astype(str).apply(lambda r: r.str.contains('youtube_305').any(),axis=1)]
print('youtube_305 rows',len(m)); print(m[[c for c in m.columns if c.startswith(('xu_','cl_'))]].drop_duplicates().to_string())
