import pandas as pd
V="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/results/failure_cases_quick/verify_dfdc"
df=pd.read_pickle(f"{V}/df.pkl"); q=pd.read_csv(f"{V}/imgstats_mine.csv").set_index("img_path"); df["lum"]=df.path.map(q.lum)
df["lq"]=pd.qcut(df.lum,4,labels=[1,2,3,4]); r=df[df.y==0]
print(r.groupby(["cl_race_vid","lq"],observed=True).agg(n=("pm","size"),v=("vid","nunique"),fpr=("pm","mean")).unstack())
