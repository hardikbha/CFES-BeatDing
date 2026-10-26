from v_common import *
df,th=load()
f=df[df.label==1].copy(); r=df[df.label==0].copy()
ids=f.cluster.str.extract(r'^0_(\d+)_(\d+)__'); print(f[ids[0]=='17'].groupby(ids[1]).size())
fp=r.pred; dfp=r.dpred; fn=~f.pred; dfn=~f.dpred
print('FP',fp.sum(),'shared',(fp&dfp).sum(),100*(fp&dfp).sum()/fp.sum(),'Ding FPR',100*dfp.mean())
print('FN',fn.sum(),'shared',(fn&dfn).sum(),100*(fn&dfn).sum()/fn.sum(),'Ding FNR',100*dfn.mean(),'ours FNR',100*fn.mean(),'ratio',dfn.mean()/fn.mean(),'ours-only',(fn&~dfn).sum())
# oracle: ding thresholded so FP count == ours
nfp=fp.sum(); ds=np.sort(r.dlogit.values)[::-1]; t=ds[nfp-1]; ofp=r.dlogit>=t
print('oracle FP count',ofp.sum(),'overlap',100*(fp&ofp).sum()/nfp)
nfn=fn.sum(); fs=np.sort(f.dlogit.values); t2=fs[nfn-1]; ofn=f.dlogit<=t2
print('oracle FN count',ofn.sum(),'overlap',100*(fn&ofn).sum()/nfn)
v=f.groupby('cluster').agg(o=('pred',lambda s:100*(1-s.mean())),d=('dpred',lambda s:100*(1-s.mean())),n=('pred','size'))
v['diff']=v.o-v.d; print(v.sort_values('diff',ascending=False).head(4))
rv=r.groupby('cluster').agg(o=('pred','sum'),d=('dpred','sum'),n=('pred','size'))
print(rv.loc['0_17__kitchen_still']); print(rv[(rv.o==0)&(rv.d==rv.n)])
