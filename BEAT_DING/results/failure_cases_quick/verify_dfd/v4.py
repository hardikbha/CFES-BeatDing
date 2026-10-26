from v_common import *
df,th=load()
L=pd.read_csv(f'{A}/DEMO_AXES/labels/labels_dfd.csv')
m=df.merge(L[['img_path','ita4','ita4_local','age4_if','if_age','ita_vid']],on='img_path')
m['scene']=m.cluster.str.extract(r'__(.+?)(?:__|$)')[0]
r=m[m.label==0]
print(r.groupby('scene').pred.agg(['mean','size']).sort_values('mean').assign(mean=lambda x:100*x['mean']))
for c in ['ita4','ita4_local','age4_if']:
    print(c); print(r.groupby(c).agg(ours=('pred','mean'),ding=('dpred','mean'),n=('pred','size'),itamed=('ita_vid','median')).assign(ours=lambda x:100*x.ours,ding=lambda x:100*x.ding))
print(r.groupby(pd.cut(r.if_age,[0,29.999,44.999,200])).pred.agg(['mean','size']))
