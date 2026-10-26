import sys, os, json, numpy as np, pandas as pd
ROOT="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CFES"
sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/scripts")
from cfes.metrics import rank_prob
from calibrate import fit_intercept
DS=ROOT+"/../DINGREPRO_LIN/results/audit_scores"
S=os.path.dirname(os.path.abspath(__file__))
SETS={"ffpp_test":"ff++","dfdc":"dfdc","celebdf":"celebdf","dfd":"dfd"}
v=np.load(f"{DS}/full_s7_ep49_ffval.npz",allow_pickle=True); bd=fit_intercept(v["logit"],v["label"])[0]
def ffpr(pr,y,part,min_real=1):
    fa=(pr[y==0]>=.5).mean(); s=0; rows={}
    for k in np.unique(part):
        r=(part==k)&(y==0); f=(part==k)&(y==1)
        if r.sum()>=min_real and f.sum()>0:
            fp=(pr[r]>=.5).mean(); s+=abs(fp-fa); rows[int(k)]=(int(r.sum()),round(100*fp,1))
    return round(100*s,1),rows
for s,dk in SETS.items():
    ref=np.load(f"{ROOT}/runs_rev4/clip_D_w0.3_s7/eval/{s}_scores.npz",allow_pickle=True)
    y=ref["label"].astype(int); b=[os.path.basename(p) for p in ref["img_path"]]
    a=pd.read_csv(f"{S}/aiface_{s}.csv").drop_duplicates('b').set_index('b').loc[b]
    sk=a['Skin Tone Group'].values; age=a['Predicted Age'].values
    out={}
    M={"Ding full s7":[(bd,np.load(f"{DS}/full_s7_ep49_{dk}.npz",allow_pickle=True)["logit"].astype(float))]}
    for arm,pat in (("A","clip_A_s{}"),("D","clip_D_w0.3_s{}")):
        M["CFES "+arm]=[(json.load(open(f"{ROOT}/runs_rev4/{pat.format(sd)}/eval/calibration.json"))["intercept_b"],
                        np.load(f"{ROOT}/runs_rev4/{pat.format(sd)}/eval/{s}_scores.npz",allow_pickle=True)["logit"].astype(float)) for sd in (7,17,27)]
    print("==",s)
    for nm,L in M.items():
        sks=[ffpr(rank_prob(z+bb),y,sk) for bb,z in L]; ags=[ffpr(rank_prob(z+bb),y,age) for bb,z in L]
        print(f"  {nm}: skin F_FPR {np.mean([x[0] for x in sks]):.1f} groups(n_real,FPR) {sks[0][1]} | age F_FPR {np.mean([x[0] for x in ags]):.1f} groups {ags[0][1]}")
