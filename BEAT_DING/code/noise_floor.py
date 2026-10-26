"""Noise floor of FF++ test intersectional F_FPR: permute group labels across REAL VIDEOS (video integrity kept),
threshold fixed at pooled real FPR 12.5% (Ding's operating point) and 5%. If a model's group effect were zero, its
i-F_FPR would be drawn from this null distribution."""
import numpy as np, os, collections
R="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
DS=R+"/agents/DINGREPRO_LIN/results/audit_scores"; RU=R+"/agents/CFES/runs_rev4"
def ld(f):
    z=np.load(f,allow_pickle=True); return z["logit"].astype(float),z["label"].astype(int),z["group"].astype(int),z["img_path"]
def vid(p):
    b=os.path.basename(str(p)); return "_".join(b.rsplit(".",1)[0].split("_")[:-1])
rng=np.random.default_rng(0)
for n,f in {"Ding repro":f"{DS}/full_s7_ep49_ff++.npz","armE w3 s7":f"{RU}/clip_E_w3_s7/eval/ffpp_test_scores.npz","CFES-D s27":f"{RU}/clip_D_w0.3_s27/eval/ffpp_test_scores.npz"}.items():
    z,y,g,p=ld(f); r=y==0; zr,gr=z[r],g[r]; vr=np.array([vid(x) for x in p[r]])
    vids=np.unique(vr); vg={v:gr[vr==v][0] for v in vids}; vidx={v:np.where(vr==v)[0] for v in vids}
    for fpr in (0.125,0.05):
        t=np.quantile(zr,1-fpr); fl=zr>=t; F0=fl.mean()
        def iF(glab):
            return 100*sum(abs(fl[glab==k].mean()-F0) for k in range(8) if (glab==k).any())
        obs=iF(gr); null=[]
        labs=np.array([vg[v] for v in vids])
        for _ in range(2000):
            perm=rng.permutation(labs); gl=np.empty(len(zr),int)
            for v,lab in zip(vids,perm): gl[vidx[v]]=lab
            null.append(iF(gl))
        null=np.array(null)
        print(f"{n:11s} FPR {100*fpr:4.1f}%: observed i-F_FPR(reals only) {obs:5.1f} | null (no group effect) median {np.median(null):5.1f}, 90% range [{np.quantile(null,.05):.1f}, {np.quantile(null,.95):.1f}] | P(null >= obs) {(null>=obs).mean():.3f}")
