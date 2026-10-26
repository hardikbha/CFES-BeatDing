import numpy as np, collections, os, re
R="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
DS=R+"/agents/DINGREPRO_LIN/results/audit_scores"; RU=R+"/agents/CFES/runs_rev4"
def ld(f):
    z=np.load(f,allow_pickle=True); return z["logit"].astype(float),z["label"].astype(int),z["group"].astype(int),z["img_path"]
def vid(p):
    b=os.path.basename(str(p)); parts=b.rsplit(".",1)[0].split("_"); return "_".join(parts[:-1])
M={"Ding repro":f"{DS}/full_s7_ep49_ff++.npz","armE w3 s7":f"{RU}/clip_E_w3_s7/eval/ffpp_test_scores.npz","CFES-D s27":f"{RU}/clip_D_w0.3_s27/eval/ffpp_test_scores.npz","CFES-D s7":f"{RU}/clip_D_w0.3_s7/eval/ffpp_test_scores.npz","armE w1 s7":f"{RU}/clip_E_w1_s7/eval/ffpp_test_scores.npz"}
for n,f in M.items():
    z,y,g,p=ld(f); t=np.quantile(z[y==0],1-0.125); pr=z>=t
    for grp,name in ((4,"nonmale,asian"),(0,"male,asian")):
        m=(g==grp)&(y==0); vids=collections.defaultdict(list)
        for pi,flag in zip(p[m],pr[m]): vids[vid(pi)].append(flag)
        rates=sorted(((np.mean(v),k,len(v)) for k,v in vids.items()),reverse=True)
        nflag=sum(1 for r,_,_ in rates if r>0.5)
        print(f"{n:11s} {name:14s} reals {m.sum()} in {len(vids)} videos; FPR {100*pr[m].mean():.1f}; videos with >50% frames flagged: {nflag}; top: "+", ".join(f"{k}:{100*r:.0f}%/{c}" for r,k,c in rates[:4]))
    # male-asian fakes TPR by video
    m=(g==0)&(y==1); vids=collections.defaultdict(list)
    for pi,flag in zip(p[m],pr[m]): vids[vid(pi)].append(flag)
    rates=sorted(((np.mean(v),k,len(v)) for k,v in vids.items()))
    print(f"{'':11s} male,asian fakes {m.sum()} in {len(vids)} videos; TPR {100*pr[m].mean():.1f}; lowest: "+", ".join(f"{k}:{100*r:.0f}%/{c}" for r,k,c in rates[:4]))
