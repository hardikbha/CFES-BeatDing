import numpy as np, pandas as pd, json, os, re
A='/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents'
def load():
    d=np.load(f'{A}/BEAT_DING/runs/BD0_s7/eval/dfd_scores.npz',allow_pickle=True)
    e=np.load(f'{A}/DINGREPRO_LIN/results/audit_scores/full_s7_ep49_dfd.npz',allow_pickle=True)
    assert (d['img_path']==e['img_path']).all() and (d['label']==e['label']).all()
    th=json.load(open(f'{A}/BEAT_DING/runs/BD0_s7/eval/bd_eval.json'))['thresholds']
    df=pd.DataFrame({k:d[k] for k in ['logit','label','group','method','cluster','img_path']})
    df['dlogit']=e['logit']; df['dgroup']=e['group']
    df['base']=df.img_path.map(os.path.basename)
    df['pred']=df.logit>=th['matched']; df['dpred']=df.dlogit>=0
    return df,th
