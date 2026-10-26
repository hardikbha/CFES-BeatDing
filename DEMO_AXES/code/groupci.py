import sys, json, numpy as np, re, os
ROOT="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CFES"
sys.path.insert(0,ROOT); from cfes.metrics import SCHEMES, rank_prob
r=f"{ROOT}/runs_rev4/clip_D_w0.3_s7/eval"; b=json.load(open(f"{r}/calibration.json"))["intercept_b"]
def wilson(k,n):
    p=k/n; z=1.96; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
out=[]
for s,scheme in (("val","eight"),("ffpp_test","eight"),("dfdc","eight"),("celebdf","six"),("dfd","six")):
    z=np.load(f"{r}/{s}_scores.npz" if s!="val" else f"{r}/val_scores.npz",allow_pickle=True)
    y=z["label"].astype(int); g=z["group"].astype(int); cl=z["cluster"] if "cluster" in z.files else np.array([re.sub(r'_\d+$','',os.path.basename(p)[:-4]) for p in z["img_path"]])
    pred=rank_prob(z["logit"].astype(float)+b)>=0.5
    rng=np.random.default_rng(0)
    for k,nm in enumerate(SCHEMES[scheme]):
        m=(g==k)&(y==0)
        if m.sum()==0: continue
        cls=np.unique(cl[m]); fpr=pred[m].mean()
        # cluster bootstrap within group
        per=[pred[m&(cl==c)] for c in cls]
        bs=[np.concatenate([per[i] for i in rng.integers(0,len(per),len(per))]).mean() for _ in range(2000)]
        lo,hi=np.percentile(bs,[2.5,97.5]); wl,wh=wilson(pred[m].sum(),m.sum())
        nf=int(((g==k)&(y==1)).sum()); nfc=len(np.unique(cl[(g==k)&(y==1)])) if nf else 0
        out.append(dict(set=s,group=nm,n_real=int(m.sum()),real_clusters=len(cls),n_fake=nf,fake_clusters=nfc,FPR=round(100*fpr,1),
            ci_cluster=[round(100*lo,1),round(100*hi,1)],width_cluster=round(100*(hi-lo),1),width_iid_wilson=round(100*(wh-wl),1)))
for o in out: print(o)
json.dump(out,open(os.path.dirname(os.path.abspath(__file__))+"/groupci.json","w"),indent=0)
