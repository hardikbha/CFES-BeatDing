# Independent verifier for DFDC failure-case claims (fresh code, raw score files only).
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
A="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents"
z=np.load(f"{A}/BEAT_DING/runs/BD0_s7/eval/dfdc_scores.npz",allow_pickle=True)
dg=np.load(f"{A}/DINGREPRO_LIN/results/audit_scores/full_s7_ep49_dfdc.npz",allow_pickle=True)
vz=np.load(f"{A}/BEAT_DING/runs/BD0_s7/eval/val_scores.npz",allow_pickle=True)
T=json.load(open(f"{A}/BEAT_DING/runs/BD0_s7/eval/bd_eval.json"))["thresholds"]
df=pd.DataFrame(dict(path=z["img_path"].astype(str),y=z["label"].astype(int),lg=z["logit"].astype(np.float64),g=z["group"].astype(int),meth=z["method"].astype(str)))
assert (dg["img_path"].astype(str)==df.path.values).all() and (dg["label"]==df.y.values).all()
df["dl"]=dg["logit"].astype(np.float64)
L=pd.read_csv(f"{A}/DEMO_AXES/labels/labels_dfdc.csv"); assert (L.img_path.values==df.path.values).all()
df["vid"]=L.video.values
for c in ["xu_race","xu_male","xu_group","cl_race","cl_male","cl_race_vid","cl_male_vid","mst3_vid","ita4","ita4_local"]: df[c]=L[c].values
assert (df.xu_group==df.g).all()
F=pd.read_csv(f"{A}/DEMO_AXES/labels_fairface/dfdc.csv").drop_duplicates("img_path").set_index("img_path")
df["race7"]=df.path.map(F.race7)
out={}
R=df.y==0; K=df.y==1
def rate(m,sub): return 100*m[sub].mean()
t=T["matched"]; df["pm"]=df.lg>=t; df["pd"]=df.dl>=0
# Claim 1
c1=dict(n=len(df),n_unique=df.path.nunique(),n_dup_paths=int(df.path.duplicated(keep=False).sum()- (df.path.duplicated(keep=False).sum()-df.path.duplicated().sum())), dup_rows=int(df.path.duplicated().sum()))
c1.update(t=t,FP=int((df.pm&R).sum()),nR=int(R.sum()),FPR=rate(df.pm,R),FN=int((~df.pm&K).sum()),nK=int(K.sum()),FNR=rate(~df.pm,K),TPR=rate(df.pm,K))
vy=vz["label"].astype(int); vl=vz["logit"].astype(np.float64)
c1["val_FPR"]=100*(vl[vy==0]>=t).mean(); c1["val_TPR"]=100*(vl[vy==1]>=t).mean()
c1["med_real_dfdc"]=float(np.median(df.lg[R])); c1["med_real_val"]=float(np.median(vl[vy==0]))
near=(df.lg-t).abs()<=10; c1["near_real"]=rate(near,R); c1["near_fake"]=rate(near,K)
for op in ["native","A1"]: c1[op]=dict(FPR=rate(df.lg>=T[op],R),TPR=rate(df.lg>=T[op],K))
c1["ding"]=dict(FPR=rate(df.pd,R),TPR=rate(df.pd,K))
out["c1"]=c1
# Claim 2
rv=df[R].groupby("vid").agg(n=("pm","size"),fp=("pm","sum")); rv["fpr"]=rv.fp/rv.n
s=rv.fp.sort_values(ascending=False); cs=s.cumsum()/s.sum()
out["c2"]=dict(n_real_vid=len(rv),vid_with_fp=int((rv.fp>0).sum()),ge50=int((rv.fpr>=.5).sum()),ge90=int((rv.fpr>=.9).sum()),
  top15_share=100*s.head(15).sum()/s.sum(),n_half=int((cs<0.5).sum()+1),n_80=int((cs<0.8).sum()+1),median_vid_fpr=100*rv.fpr.median())
# Claim 4
c4={}
for col in ["ita4","ita4_local","mst3_vid","cl_race","cl_male","race7"]:
    gg=df[R].groupby(col,dropna=False); c4[col]={str(k):[round(100*v.pm.mean(),2),len(v)] for k,v in gg}
out["c4"]=c4
# Claim 5: F_FPR
def ffpr(y,pred,grp):
    allf=pred[y==0].mean(); tot=0; per={}
    for gid in np.unique(grp):
        m=grp==gid
        if (y[m]==0).sum()==0 or (y[m]==1).sum()==0: continue
        f=pred[m&(y==0)].mean(); per[int(gid)]=f; tot+=abs(f-allf)
    return 100*tot,per,allf
y=df.y.values; pm=df.pm.values; gx=df.g.values
base,per,allf=ffpr(y,pm,gx)
NM=["M-Asian","M-White","M-Black","M-Others","F-Asian","F-White","F-Black","F-Others"]
c5=dict(xu_iFFPR=base,dev={NM[k]:round(100*(v-allf),2) for k,v in per.items()},grpFPR={NM[k]:round(100*v,2) for k,v in per.items()})
# permutation null: shuffle video->group among real videos
vids=df.vid.values
rvg=df[R].groupby("vid").g.first()
assert (df[R].groupby("vid").g.nunique()==1).all()
rng=np.random.default_rng(12345)
ridx=np.where(R)[0]; rvid=vids[ridx]; uv=rvg.index.values; inv=pd.Index(uv).get_indexer(rvid)
def null_dist(gvals,n=2000,seed=1):
    r=np.random.default_rng(seed); res=[]
    for _ in range(n):
        gp=gx.copy(); gp[ridx]=r.permutation(gvals)[inv]; res.append(ffpr(y,pm,gp)[0])
    return np.array(res)
nl=null_dist(rvg.values)
c5.update(null_med=float(np.median(nl)),null_p5=float(np.percentile(nl,5)),null_p95=float(np.percentile(nl,95)),p=float((nl>=base).mean()))
# greedy
def greedy(gvec,maxsteps=40):
    keep=np.ones(len(df),bool); b0=ffpr(y,pm,gvec)[0]; cur=b0; steps=0
    rvl=list(uv)
    while cur>b0/2 and steps<maxsteps:
        best=None
        for v in rvl:
            k2=keep&~(vids==v)
            val=ffpr(y[k2],pm[k2],gvec[k2])[0]
            if best is None or val<best[0]: best=(val,v)
        cur=best[0]; keep&=~(vids==best[1]); rvl.remove(best[1]); steps+=1
    return steps,b0
st,_=greedy(gx); c5["greedy_halve_xu"]=st
rr=[]
r2=np.random.default_rng(7)
for i in range(20):
    gp=gx.copy(); gp[ridx]=r2.permutation(rvg.values)[inv]; rr.append(greedy(gp)[0])
c5["greedy_random"]=dict(median=float(np.median(rr)),min=int(min(rr)),max=int(max(rr)),all=rr)
# F-Black top-10 drop (by FP count and by FPR)
fb=df[R&(df.g==6)]; fbv=fb.groupby("vid").agg(fp=("pm","sum"),n=("pm","size"),fpr=("pm","mean"))
c5["fblack_nvid"]=len(fbv); c5["fblack_fpr"]=100*fb.pm.mean()
for key in ["fp","fpr"]:
    top=fbv.sort_values(key,ascending=False).head(10).index
    c5[f"fblack_after_drop10_by_{key}"]=100*fb[~fb.vid.isin(top)].pm.mean()
# cleaner groups
rank={"asian":0,"white":1,"black":2,"others":3}
gc=(4*(df.cl_male_vid!=1).astype(int)+df.cl_race_vid.map(rank)).values
assert not np.isnan(gc.astype(float)).any()
df["gc"]=gc
vc=df[R].groupby("vid").agg(gx=("g","first"),gc=("gc","first"))
c5["vid_change"]=int((vc.gx!=vc.gc).sum())
top15=rv.fp.sort_values(ascending=False).head(15).index
c5["top15_change"]=int((vc.loc[top15].gx!=vc.loc[top15].gc).sum())
bc,perc,allc=ffpr(y,pm,gc); c5["cl_iFFPR"]=bc; c5["cl_grpFPR"]={NM[k]:round(100*v,2) for k,v in perc.items()}
assert (df[R].groupby("vid").gc.nunique()==1).all()
rvc=df[R].groupby("vid").gc.first().reindex(uv).values
r=np.random.default_rng(3); nc=[]
for _ in range(2000):
    gp=gc.copy(); gp[ridx]=r.permutation(rvc)[inv]; nc.append(ffpr(y,pm,gp)[0])
nc=np.array(nc); c5.update(cl_null_med=float(np.median(nc)),cl_null_p95=float(np.percentile(nc,95)),cl_p=float((nc>=bc).mean()))
c5["ding_cl_iFFPR"]=ffpr(y,df.pd.values,gc)[0]; c5["ding_xu_iFFPR"]=ffpr(y,df.pd.values,gx)[0]
out["c5"]=c5
# Claim 6
fn=~df.pm&K
c6=dict(FN=int(fn.sum()),FWhite_FN=int((fn&(df.g==5)).sum()),FWhite_share_FN=100*(fn&(df.g==5)).sum()/fn.sum(),FWhite_share_fake=100*(K&(df.g==5)).sum()/K.sum(),
  FNR_FWhite=rate(~df.pm,K&(df.g==5)),FNR_MWhite=rate(~df.pm,K&(df.g==1)),FNR_female=rate(~df.pm,K&(df.xu_male!=1)),FNR_male=rate(~df.pm,K&(df.xu_male==1)))
fv=df[K].groupby("vid").agg(n=("pm","size"),fn=("pm",lambda s:(~s).sum()),g=("g","first"),dfn=("pd",lambda s:(~s).sum()))
fv["fnr"]=fv.fn/fv.n; fv["dfnr"]=fv.dfn/fv.n
top=fv.sort_values(["fn","fnr"],ascending=False).head(15)
c6.update(n_fake_vid=len(fv),ge50=int((fv.fnr>=.5).sum()),all=int((fv.fnr==1).sum()),ding_ge50=int((fv.dfnr>=.5).sum()),
  top15_FWhite_by_count=int((top.g==5).sum()),top15_FWhite_by_fnr=int((fv.sort_values("fnr",ascending=False).head(15).g==5).sum()),
  methods=df[K].meth.value_counts().to_dict())
out["c6"]=c6
# Claim 7
a=df[K&(df.xu_race=="asian")]
out["c7"]=dict(vids=int(a.vid.nunique()),rows=len(a),FNR=100*(~a.pm).mean(),
  FA=[int(a[a.g==4].vid.nunique()),int((a.g==4).sum()),100*(~a[a.g==4].pm).mean()],
  MA=[int(a[a.g==0].vid.nunique()),int((a.g==0).sum()),100*(~a[a.g==0].pm).mean()])
# Claim 8
ofp=df.pm&R; dfp=df.pd&R; ofn=~df.pm&K; dfn=~df.pd&K
c8=dict(fp_shared=100*(ofp&dfp).sum()/ofp.sum(),chance_fp=rate(df.pd,R),fn_shared=100*(ofn&dfn).sum()/ofn.sum(),chance_fn=rate(~df.pd,K),
  only_ours_fp=int((ofp&~dfp).sum()),only_ours_fn=int((ofn&~dfn).sum()),only_ding_fp=int((dfp&~ofp).sum()),only_ding_fn=int((dfn&~ofn).sum()),
  rho_real=float(spearmanr(df.lg[R],df.dl[R])[0]),rho_fake=float(spearmanr(df.lg[K],df.dl[K])[0]))
rvv=df[R].groupby("vid").agg(o=("pm","mean"),d=("pd","mean")); c8["vid_fp_ours_gt50"]=int(((rvv.o-rvv.d)>=.5).sum()); c8["vid_fp_ding_gt50"]=int(((rvv.d-rvv.o)>=.5).sum())
fvv=df[K].groupby("vid").agg(o=("pm",lambda s:(~s).mean()),d=("pd",lambda s:(~s).mean())); c8["vid_fn_ours_gt50"]=int(((fvv.o-fvv.d)>=.5).sum()); c8["vid_fn_ding_gt50"]=int(((fvv.d-fvv.o)>=.5).sum())
out["c8"]=c8
# Claim 9
u=df.drop_duplicates("path")
tfp=u[(u.y==0)].sort_values("lg",ascending=False).head(12)
tfn=u[(u.y==1)].sort_values("lg").head(12)
out["c9"]=dict(fp_vids=tfp.vid.value_counts().to_dict(),fp_top=[tfp.vid.iloc[0],tfp.lg.iloc[0]],fp_7434=tfp[tfp.vid=="0_7434"].lg.round(2).tolist(),fp_7434_g=NM[int(tfp[tfp.vid=="0_7434"].g.iloc[0])] if (tfp.vid=="0_7434").any() else None,
  fn_vids=tfn.vid.value_counts().to_dict(),fn_4680=tfn[tfn.vid=="0_4680"].lg.round(2).tolist(),fn_4680_g=NM[int(tfn[tfn.vid=="0_4680"].g.iloc[0])] if (tfn.vid=="0_4680").any() else None,
  fn_4680_clgrp=NM[int(tfn[tfn.vid=="0_4680"].gc.iloc[0])] if (tfn.vid=="0_4680").any() else None)
# contact sheet items: recompute selection (top-24 unique videos)
sfp=u[(u.y==0)&u.pm].sort_values("lg",ascending=False).drop_duplicates("vid").head(24)
sfn=u[(u.y==1)&~u.pm].sort_values("lg").drop_duplicates("vid").head(24)
out["sheets"]=dict(fp_ding_wrong=int(sfp.pd.sum()),fn_ding_wrong=int((~sfn.pd).sum()),fn_white_women_cl=int((sfn.gc==5).sum()),fn_white_women_xu=int((sfn.g==5).sum()),
  fp_paths=sfp.path.tolist(),fn_paths=sfn.path.tolist())
df.to_pickle(__file__.replace("v_main.py","df.pkl"))
json.dump(out,open(__file__.replace("v_main.py","v_main.json"),"w"),indent=1,default=str)
print(json.dumps({k:v for k,v in out.items() if k!="sheets"},indent=1,default=str)); print({k:v for k,v in out["sheets"].items() if "paths" not in k})
