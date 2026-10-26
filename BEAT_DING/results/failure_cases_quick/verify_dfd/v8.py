from v_common import *
df,th=load()
r=df[df.label==0]; f=df[df.label==1]
fp=r[r.pred]; fn=f[~f.pred]
i=fp.logit.idxmax(); print('max FP',fp.logit.max(),fp.base[i])
print('FP logit>=0',(fp.logit>=0).sum(),'FP >=A1',(fp.logit>=th['A1']).sum())
j=fn.logit.idxmin(); print('min FN',fn.logit.min(),fn.base[j])
# sheet: one highest-confidence error per video
fpt=fp.loc[fp.groupby('cluster').logit.idxmax()].sort_values('logit',ascending=False)
fnt=fn.loc[fn.groupby('cluster').logit.idxmin()].sort_values('logit')
print('FP videos',len(fpt),'FN videos',len(fnt))
for name,t,k in [('fp',fpt,'dpred_wrong_fp'),('fn',fnt,'')]:
    top=t.head(24)
    right = (~top.dpred).sum() if name=='fp' else top.dpred.sum()
    print(name,'top24 Ding right',right)
S=pd.read_csv(f'{A}/BEAT_DING/results/failure_cases/dfd/fp_sheet_tiles.csv'); T=pd.read_csv(f'{A}/BEAT_DING/results/failure_cases/dfd/fn_sheet_tiles.csv')
print(S.columns.tolist()); print(len(S),len(T))
from PIL import Image
for nm,X,ref in [('fp',S,fpt),('fn',T,fnt)]:
    X=X.rename(columns={'logit':'s_logit'}).copy(); X['base']=X['img_path'].map(os.path.basename)
    mm=X.merge(df[['base','logit','dlogit','label','cluster']],on='base',how='left')
    print(nm,'all found',mm.logit.notna().all(),'tiles in my top24', mm.base.isin(ref.head(24).base).sum(),
          'ding right', int(((mm.dlogit<0)&(mm.label==0)).sum()+((mm.dlogit>=0)&(mm.label==1)).sum()),
          'max|logit diff|',np.abs(mm.s_logit-mm.logit).max(),'max|ding diff|',np.abs(mm.ding_logit-mm.dlogit).max())
    for p in mm.img_path.head(3):
        im=Image.open(p); print('   ',os.path.basename(p),im.size,im.mode,os.path.getsize(p))
