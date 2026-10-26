from v_common import *
df,th=load()
names=["M-White","M-Black","M-Others","F-White","F-Black","F-Others"]
def iffpr(d):
    y=d.label.values; p=d.pred.values; g=d.group.values
    fall=p[y==0].mean(); tot=0; per={}
    for gi in range(6):
        m=g==gi
        if m.sum()==0 or len(np.unique(y[m]))<2: continue
        fg=p[m&(y==0)].mean(); per[names[gi]]=fg; tot+=abs(fg-fall)
    return 100*tot,per,fall
v,per,fall=iffpr(df); print('iF_FPR',v,{k:round(100*x,2) for k,x in per.items()},100*fall)
print(df.groupby(['group','label']).size().unstack())
fb=df[(df.group==4)&(df.label==0)]
print('F-Black real videos', fb.groupby('cluster').pred.agg(['size','mean']), len(fb), fb.pred.mean())
# also cross-check with cfes.metrics
import sys; sys.path.insert(0,f'{A}/BEAT_DING')
from cfes.metrics import fairness_block
fbk=fairness_block(df.label.values, df.logit.values, df.group.values,'six',th['matched'])
print('cfes iF_FPR',fbk['intersection']['F_FPR'], fbk['intersection']['groups_skipped_single_class'])
# removal
vids=df[df.label==0].cluster.unique()
d1=df[df.cluster!='0_19__podium_speech_happy']; print('remove podium', iffpr(d1)[0])
# best single, then greedy
cur=df.copy(); seq=[]
for k in range(10):
    best=None
    for c in cur[cur.label==0].cluster.unique():
        val=iffpr(cur[cur.cluster!=c])[0]
        if best is None or val<best[0]: best=(val,c)
    cur=cur[cur.cluster!=best[1]]; seq.append(best); print(k+1,best)
# bootstrap
rng=np.random.default_rng(0)
vr=df[df.label==0].cluster.unique(); vf=df[df.label==1].cluster.unique()
gb={c:g for c,g in df.groupby('cluster')}
bs=[]
for b in range(1000):
    s=list(rng.choice(vr,len(vr)))+list(rng.choice(vf,len(vf)))
    bs.append(iffpr(pd.concat([gb[c] for c in s]))[0])
print('boot CI',np.percentile(bs,[2.5,97.5]))
