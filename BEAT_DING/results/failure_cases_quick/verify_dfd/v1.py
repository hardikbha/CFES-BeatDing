from v_common import *
df,th=load()
r=df[df.label==0]; f=df[df.label==1]
print('groups same as ding', (df.group==df.dgroup).all())
print('n real',len(r),'n fake',len(f))
for k,t in th.items(): print(k,t,'FP',(r.logit>=t).sum(),100*(r.logit>=t).mean(),'TPR',100*(f.logit>=t).mean())
fp=r[r.pred]
pv=r.groupby('cluster').pred.agg(['sum','size','mean'])
print('real videos',len(pv),'>=1FP',(pv['sum']>=1).sum(),'>=50%',(pv['mean']>=0.5).sum())
s=pv['sum'].sort_values(ascending=False).cumsum()/pv['sum'].sum()
print('videos to 50% FPs', int((s<0.5).sum())+1)
print('real median logit',r.logit.median())
# actor = first id in name of real: 0_XX__scene
r=r.copy(); r['actor']=r.cluster.str.extract(r'^0_(\d+)__')[0]
print('actor parse na',r.actor.isna().sum(), 'method==actor', (r.method.astype(str)==r.actor).mean())
am=r.groupby('actor').logit.median(); print('actor median range',am.min(),am.max(), len(am))
print('Ding FPR',100*r.dpred.mean(),'TPR',100*f.dpred.mean())
print('Ours FPR',100*r.pred.mean(),'TPR',100*f.pred.mean())
