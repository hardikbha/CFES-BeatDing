# Profile/align_scale/master path for TRAIN/VAL fakes only (rows matched to the train/val manifests).
import pandas as pd, glob
from multiprocessing import Pool
OUT='/home/sachin.chaudhary/ces_fl_runs/ff2_source_diag_20261106'
m=pd.concat([pd.read_csv('/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/%s.csv'%s,usecols=['image','label','source_ids','method']).assign(sp=s) for s in ['train','val']])
KEEP=set(m.image[m.label==1])
WANT=['item_id','method','tier','profile','align_scale','fake_path','master_path']
def one(f):
    d=pd.read_csv(f,low_memory=False); d=d[[c for c in WANT if c in d.columns]]
    return d[d.fake_path.isin(KEEP)]
if __name__=='__main__':
    fs=sorted(glob.glob('/home/sachin.chaudhary/ICLR_2027/fairforge_scaleup/fin_records/*/*.csv'))
    with Pool(8) as p: L=p.map(one,fs,chunksize=8)
    d=pd.concat(L).drop_duplicates('fake_path'); d=d.merge(m[['image','sp','source_ids']],left_on='fake_path',right_on='image')
    d['src']=d.source_ids.str.split(':').str[0]
    d.to_pickle(OUT+'/_fin_trval_PRIVATE.pkl')
    print(len(d),'of',len(KEEP),'fakes matched')
    print(pd.crosstab([d.sp,d.src],[d.tier,d.profile.fillna('NA')]))
    print(d.groupby(['sp','src','profile']).align_scale.describe().round(3).to_string())
    print(d[d.sp=='train'].groupby(['method','src']).align_scale.median().unstack().round(3).to_string())
