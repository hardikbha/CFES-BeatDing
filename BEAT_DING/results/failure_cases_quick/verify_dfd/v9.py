from v_common import *
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
df,th=load()
df['a']=df.cluster.str.extract(r'^0_(\d+)_')[0]
acts=sorted(set(df[df.label==0].a)&set(df[df.label==1].a)); print(acts)
rows=[]
for a in acts:
    g=df[df.a==a]; r=g[g.label==0]; f=g[g.label==1]
    rows.append(dict(a=a,FPR=r.pred.mean(),FNR=1-f.pred.mean(),auc=roc_auc_score(g.label,g.logit),dauc=roc_auc_score(g.label,g.dlogit)))
R=pd.DataFrame(rows); print(R.round(4))
print(spearmanr(R.FPR,R.FNR))
s=df[df.a.isin(acts)]
print('mean within AUC',100*R.auc.mean(),'pooled',100*roc_auc_score(s.label,s.logit),'ding',100*R.dauc.mean(),100*roc_auc_score(s.label,s.dlogit))
print('full-set AUC ours',100*roc_auc_score(df.label,df.logit),'ding',100*roc_auc_score(df.label,df.dlogit))
