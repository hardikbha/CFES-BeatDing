from v_common import *
import hashlib
from PIL import Image
df,th=load()
L=pd.read_csv(f'{A}/DEMO_AXES/labels/labels_dfd.csv')
m=df.merge(L[['img_path','cl_male']],on='img_path')
m['scene']=m.cluster.str.extract(r'__(.+?)(?:__|$)')[0]
ids=m.cluster.str.extract(r'^0_(\d+)(?:_(\d+))?__')
m['a1']=ids[0]; m['a2']=ids[1]
f=m[m.label==1].copy(); r=m[m.label==0].copy()
print('fake a2 missing',f.a2.isna().sum(), 'method==a1', (f.method.astype(str)==f.a1).mean())
f['miss']=~f.pred; f['dmiss']=~f.dpred
pr=f.groupby(['a1','a2']).miss.agg(['mean','size']).sort_values('mean',ascending=False); print(pr.head(5))
print('actor16 (first id)',f[f.a1=='16'].miss.mean(), 'actor16 any', f[(f.a1=='16')|(f.a2=='16')].miss.mean())
print(f.groupby('a1').miss.mean().sort_values())
sc=f.groupby('scene').miss.agg(['mean','size']).sort_values('mean'); print(sc)
am=r.groupby('a1').cl_male.mean()
def mm(row):
    s=[am.get(row.a1)] + ([am.get(row.a2)] if isinstance(row.a2,str) else [])
    if any(x is None or pd.isna(x) for x in s): return np.nan
    exp={int(round(x)) for x in s}
    return float(row.cl_male not in exp)
f['mm']=f.apply(mm,axis=1)
print('actor male share',am.round(3).to_dict())
for v in [1.0,0.0]:
    q=f[f.mm==v]; print('mismatch',v,len(q),q.cluster.nunique(),'FNR',100*q.miss.mean(),'DingFNR',100*q.dmiss.mean())
print('unscored',f.mm.isna().sum())
# md5 on reals
r['md5']=[hashlib.md5(np.asarray(Image.open(p).convert('RGB')).tobytes()).hexdigest() for p in r.img_path]
dup=r.groupby('md5').filter(lambda g:g.cluster.nunique()>1)
print('dup real frames',len(dup),'videos',dup.cluster.nunique(), sorted(dup.cluster.unique()), 'actors',sorted(dup.a1.unique()))
