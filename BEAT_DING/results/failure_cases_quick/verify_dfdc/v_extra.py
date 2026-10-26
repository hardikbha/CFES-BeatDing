import pandas as pd, numpy as np, json
V="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/results/failure_cases_quick/verify_dfdc"
df=pd.read_pickle(f"{V}/df.pkl"); R=df.y==0; K=df.y==1
o={}
rv=df[R].groupby("vid").agg(o=("pm","mean"),d=("pd","mean"),n=("pm","size"))
fv=df[K].groupby("vid").agg(o=("pm",lambda s:(~s).mean()),d=("pd",lambda s:(~s).mean()))
for eps in [0,1e-9]:
    o[f"eps{eps}"]=[int(((rv.o-rv.d)>=.5-eps).sum()),int(((rv.d-rv.o)>=.5-eps).sum()),int(((fv.o-fv.d)>=.5-eps).sum()),int(((fv.d-fv.o)>=.5-eps).sum())]
o["fp_exact_half_ours"]=int(np.isclose(rv.o-rv.d,.5).sum()); o["fp_exact_half_ding"]=int(np.isclose(rv.d-rv.o,.5).sum())
o["fn_exact_half_ours"]=int(np.isclose(fv.o-fv.d,.5).sum()); o["fn_exact_half_ding"]=int(np.isclose(fv.d-fv.o,.5).sum())
# image quality
q=pd.read_csv(f"{V}/imgstats_mine.csv").set_index("img_path")
for c in ["lum","lstd","lap","sat"]: df[c]=df.path.map(q[c])
assert df[["lum","lstd","lap","sat"]].notna().all().all()
for c in ["lum","lstd","lap","sat"]:
    df[c+"_q"]=pd.qcut(df[c],4,labels=[1,2,3,4])
    o[c+"_FPR"]=(100*df[R].groupby(c+"_q",observed=True).pm.mean()).round(2).to_dict()
    o[c+"_FNR"]=(100*(~df[K].pm).groupby(df[K][c+"_q"],observed=True).mean()).round(2).to_dict()
t=(100*df[R].groupby(["cl_race_vid","lum_q"],observed=True).pm.mean()).round(1).unstack()
o["race_x_bright_vid"]=t.to_dict(orient="index")
t=(100*df[R].groupby(["cl_race","lum_q"],observed=True).pm.mean()).round(1).unstack()
o["race_x_bright_frame"]=t.to_dict(orient="index")
o["med_bright_real_by_cl_race_vid"]=df[R].groupby("cl_race_vid").lum.median().round(1).to_dict()
o["med_bright_real_by_cl_race"]=df[R].groupby("cl_race").lum.median().round(1).to_dict()
# compare with analyst's image_quality.csv
a=pd.read_csv("/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/results/failure_cases/dfdc/image_quality.csv").set_index("img_path").reindex(q.index)
o["max_absdiff_vs_analyst"]={m:float((q[m]-a[n]).abs().max()) for m,n in [("lum","lum_mean"),("lstd","lum_std"),("lap","sharp_lapvar"),("sat","sat_mean")]}
print(json.dumps(o,indent=1,default=str)); json.dump(o,open(f"{V}/v_extra.json","w"),indent=1,default=str)
