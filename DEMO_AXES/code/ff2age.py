import pandas as pd, numpy as np
R="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
rm=pd.read_csv(R+"/agents/T_scaleup/real_tiers/data/REAL_MANIFEST.csv",usecols=['path','tier','person_id','race','gender','age','age_source','age_dataset','split','p_adult','light_L_face','chroma'],dtype=str,low_memory=False); [rm.__setitem__(c,pd.to_numeric(rm[c],errors='coerce')) for c in ['age','p_adult','light_L_face','chroma']]
print(rm.age_source.value_counts().to_dict()); print(rm.age_dataset.value_counts(dropna=False).head(12).to_dict())
t=pd.read_csv(R+"/data_ding_schema/fairforge2_full/test.csv")
G=['M-A','M-W','M-B','M-O','F-A','F-W','F-B','F-O']
t['grp']=t.intersec_label.map(dict(enumerate(G)))
tr=t[t.label==0].merge(rm,left_on='img_path',right_on='path',how='left')
print("test reals",len(tr),"matched age",tr.age.notna().sum())
bins=[0,30,45,65,200]; lab=['20-29','30-44','45-64','65+']
tr['ab']=pd.cut(tr.age,bins,right=False,labels=lab)
ct=pd.crosstab(tr.grp,tr.ab); ct['n']=ct.sum(1); print("TEST reals InsightFace age bins:\n",ct.to_string())
pp=tr.drop_duplicates('person_id'); ctp=pd.crosstab(pp.grp,pp.ab); print("TEST real PERSONS by age bin:\n",ctp.to_string())
print("median age by group",tr.groupby('grp').age.median().to_dict())
print("light_L_face by group (median)",tr.groupby('grp').light_L_face.median().round(1).to_dict())
# all-splits reals
a=rm.copy(); a['ab']=pd.cut(a.age,bins,right=False,labels=lab); print("ALL reals in REAL_MANIFEST age bins:",a.ab.value_counts().to_dict())
fc=a[a.tier=='C']; print("tier C FairFace age_dataset:",fc.age_dataset.value_counts().to_dict())
