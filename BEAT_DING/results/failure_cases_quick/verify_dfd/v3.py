from v_common import *
df,th=load()
L=pd.read_csv(f'{A}/DEMO_AXES/labels/labels_dfd.csv'); F=pd.read_csv(f'{A}/DEMO_AXES/labels_fairface/dfd.csv')
print(len(L),len(F), L.img_path.isin(df.img_path).all())
m=df.merge(L,on='img_path',how='left',suffixes=('','_L')).merge(F[['img_path','vid_sex','sex']],on='img_path',how='left')
print('xu group == npz group', (m.xu_group==m.group).mean())
a19=m[m.person=='id:19']
print('person 19 rows',len(a19), a19.label.value_counts().to_dict())
pv=a19.groupby('video').agg(lab=('label','first'),xu=('xu_male','first'),xur=('xu_race','first'),clv=('cl_male_vid','first'),cl=('cl_male','mean'))
print(pv)
# also actor19 as real videos named 0_19__
r=m[m.label==0]
r19=r[r.cluster.str.startswith('0_19__')]; print(r19.groupby('cluster').agg(xu=('xu_male','first'),xur=('xu_race','first'),clv=('cl_male_vid','first'),cl=('cl_male','mean'),n=('pred','size')))
def fpr(col):
    out={}
    for v,g in r.groupby(col): out[v]=(round(100*g.pred.mean(),2),len(g))
    return out
print('xu_male',fpr('xu_male')); print('cl_male',fpr('cl_male')); print('cl_male_vid',fpr('cl_male_vid')); print('vid_sex',fpr('vid_sex')); print('sex frame',fpr('sex'))
print(r.cl_male.isna().sum(), r.vid_sex.isna().sum())
