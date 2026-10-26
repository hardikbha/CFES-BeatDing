import json, numpy as np, pandas as pd
exec(open(__file__.replace("verify_extra.py","verify_celebdf.py")).read().split('pr("\\n==== CLAIM 1 ====")')[0])
t=T["matched"]; p=(df.logit>=t).values
g=df.group.values; base,_=ifpr(p,g,y)
rv=df[R].assign(p=p[R]).groupby("vid").agg(fp=("p","sum"),n=("p","size"),gn=("gname","first")).sort_values("fp",ascending=False)
for i,(v,r) in enumerate(rv.head(20).iterrows()):
    keep=(df.vid!=v).values
    tot,_=ifpr(p[keep],g[keep],y[keep]); print(i+1,v,r.gn,int(r.fp),int(r.n),round(tot-base,3))
vv=df[R].groupby("vid").agg(xm=("xu_male","first"),xr=("xu_race","first"),cm=("cl_male_vid","first"),cr=("cl_race_vid","first"),o=("logit",lambda s:(s>=t).mean()))
print(vv[(vv.xm!=vv.cm)|(vv.xr!=vv.cr)])
