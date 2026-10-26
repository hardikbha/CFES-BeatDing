import json, numpy as np, pandas as pd, os, re
R='/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009'
E=f'{R}/agents/BEAT_DING/runs/BD0_s7/eval'
th=json.load(open(f'{E}/bd_eval.json'))['thresholds']
t=th['matched'] if not isinstance(th['matched'],dict) else th['matched']['t']
G8={0:'M-Asian',1:'M-White',2:'M-Black',3:'M-Others',4:'F-Asian',5:'F-White',6:'F-Black',7:'F-Others'}
G6={0:'M-White',1:'M-Black',2:'M-Others',3:'F-White',4:'F-Black',5:'F-Others'}
DK={'ffpp_test':'ff++','dfdc':'dfdc','celebdf':'celebdf','dfd':'dfd'}
def vid(p):
    b=os.path.basename(p)[:-4]; parts=b.split('_')
    return '_'.join(parts[1:-1]) if len(parts)>2 else b
out={}
for s,dk in DK.items():
    d=np.load(f'{E}/{s}_scores.npz',allow_pickle=True)
    z=d['logit'].astype(float); y=d['label']; g=d['group']; m=d['method']; P=d['img_path']
    dd=np.load(f'{R}/agents/DINGREPRO_LIN/results/audit_scores/full_s7_ep49_{dk}.npz',allow_pickle=True)
    assert all(os.path.basename(a)==os.path.basename(b) for a,b in zip(P[:2000],dd['img_path'][:2000]))
    ours=z>=t; ding=dd['logit'].astype(float)>=0
    df=pd.DataFrame({'y':y,'g':[(G6 if s in ('celebdf','dfd') else G8).get(int(x),str(x)) for x in g],'m':m,'v':[vid(p) for p in P],'ours':ours,'ding':ding,'z':z})
    R0=df[df.y==0]; F1=df[df.y==1]
    fp=R0[R0.ours]; fn=F1[~F1.ours]
    o={'n_real':len(R0),'n_fake':len(F1),'FPR':round(100*len(fp)/len(R0),1),'TPR':round(100*F1.ours.mean(),1),
       'Ding_FPR':round(100*R0.ding.mean(),1),'Ding_TPR':round(100*F1.ding.mean(),1),
       'FP_shared_with_Ding_%':round(100*fp.ding.mean(),1) if len(fp) else None,
       'FN_shared_with_Ding_%':round(100*(~fn.ding).mean(),1) if len(fn) else None,
       'FPR_by_group':R0.groupby('g').ours.agg(['mean','size']).assign(mean=lambda x:(100*x['mean']).round(1)).to_dict('index'),
       'Ding_FPR_by_group':(100*R0.groupby('g').ding.mean()).round(1).to_dict(),
       'TPR_by_method':(100*F1.groupby('m').ours.mean()).round(1).to_dict(),
       'Ding_TPR_by_method':(100*F1.groupby('m').ding.mean()).round(1).to_dict()}
    vv=R0.groupby('v').agg(n=('ours','size'),fp=('ours','sum'),g=('g','first'),ding_fp=('ding','sum'))
    vv['rate']=(100*vv.fp/vv.n).round(0)
    o['top_FP_videos']=vv.sort_values('fp',ascending=False).head(8).reset_index().to_dict('records')
    o['FP_frames_in_top5_videos_%']=round(100*vv.sort_values('fp',ascending=False).fp.head(5).sum()/max(1,len(fp)),1)
    fv=F1.groupby('v').agg(n=('ours','size'),miss=('ours',lambda x:(~x).sum()),g=('g','first'),m=('m','first'),ding_miss=('ding',lambda x:(~x).sum()))
    o['top_FN_videos']=fv.sort_values('miss',ascending=False).head(8).reset_index().to_dict('records')
    out[s]=o
json.dump(out,open(os.path.dirname(os.path.abspath(__file__))+'/quick.json','w'),indent=1,default=str)
for s,o in out.items():
    print(f"\n=== {s}: real {o['n_real']} fake {o['n_fake']} | ours FPR {o['FPR']} TPR {o['TPR']} | Ding FPR {o['Ding_FPR']} TPR {o['Ding_TPR']} | FP also Ding-FP {o['FP_shared_with_Ding_%']}% | FN also Ding-miss {o['FN_shared_with_Ding_%']}% | top5 videos hold {o['FP_frames_in_top5_videos_%']}% of FPs")
    print(' FPR by group (ours/Ding/n):', {k:(v['mean'],o['Ding_FPR_by_group'].get(k),v['size']) for k,v in o['FPR_by_group'].items()})
    print(' TPR by method (ours/Ding):', {k:(v,o['Ding_TPR_by_method'].get(k)) for k,v in o['TPR_by_method'].items()})
    print(' top FP videos:', [(r['v'],r['g'],int(r['fp']),int(r['n']),int(r['ding_fp'])) for r in o['top_FP_videos'][:6]])
    print(' top FN videos:', [(r['v'],r['g'],r['m'],int(r['miss']),int(r['n']),int(r['ding_miss'])) for r in o['top_FN_videos'][:5]])
