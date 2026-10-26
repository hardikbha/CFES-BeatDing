import sys, os, json, numpy as np, pandas as pd
ROOT="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CFES"
sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/scripts")
from cfes.metrics import SCHEMES, fairness_block, rank_prob
from calibrate import fit_intercept
DS=ROOT+"/../DINGREPRO_LIN/results/audit_scores"
P="/home/sachin.chaudhary/models_external/robust_fairness_withNoisy_labels/dataset"
SETS={"ffpp_test":("ff++","eight","ff++/test_noise.csv"),"dfdc":("dfdc","eight","dfdc/test_noise.csv"),
      "celebdf":("celebdf","six","celebdf/test_noise.csv"),"dfd":("dfd","six","dfd/test_noise.csv")}
RACE={0:'asian',1:'white',2:'black',3:'others'}
B=int(os.environ.get("B",400))
def clean_groups(paths,scheme,pf):
    p=pd.read_csv(os.path.join(P,pf)); p['b']=p['Image Path'].map(os.path.basename); p=p.drop_duplicates('b').set_index('b')
    b=[os.path.basename(x) for x in paths]; q=p.loc[b]
    names=SCHEMES[scheme]; out=[]; age=q['Ground Truth Age'].values
    for gg,rr in zip(q['Ground Truth Gender'].values,q['Ground Truth Race'].values):
        r=RACE[rr]
        if scheme=="six" and r=="asian": r="others"
        out.append(names.index(("male" if gg==1 else "nonmale")+","+r))
    return np.array(out),age
def metr(y,z,g,scheme,b,age):
    pr=rank_prob(z+b); f=fairness_block(y,pr,g,scheme,0.5)
    o={"i":f["intersection"].get("F_FPR",np.nan),"g":f["gender"].get("F_FPR",np.nan),"r":f["race"].get("F_FPR",np.nan)}
    # age F_FPR over young/mid/senior (3=others excluded), Ding eq.8 form
    m=age<3; fpr_all=(pr[y==0]>=0.5).mean(); s=0; k=0
    for a in (0,1,2):
        mm=(age==a)&(y==0)
        if mm.sum()>0 and ((age==a)&(y==1)).sum()>0: s+=abs((pr[mm]>=0.5).mean()-fpr_all); k+=1
    o["age"]=100*s if k>=2 else np.nan
    return o
def models():
    M={}
    v=np.load(f"{DS}/full_s7_ep49_ffval.npz",allow_pickle=True); b=fit_intercept(v["logit"],v["label"])[0]
    M["Ding full s7"]=[(b,{s:np.load(f"{DS}/full_s7_ep49_{dk}.npz",allow_pickle=True)["logit"].astype(float) for s,(dk,_,_) in SETS.items()})]
    for arm,pat in (("A","clip_A_s{}"),("D","clip_D_w0.3_s{}")):
        L=[]
        for sd in (7,17,27):
            r=f"{ROOT}/runs_rev4/{pat.format(sd)}/eval"
            bb=json.load(open(f"{r}/calibration.json"))["intercept_b"]
            L.append((bb,{s:np.load(f"{r}/{s}_scores.npz",allow_pickle=True)["logit"].astype(float) for s in SETS}))
        M["CFES "+arm]=L
    return M
M=models(); res={}
for s,(dk,scheme,pf) in SETS.items():
    ref=np.load(f"{ROOT}/runs_rev4/clip_D_w0.3_s7/eval/{s}_scores.npz",allow_pickle=True)
    y=ref["label"].astype(int); gx=ref["group"].astype(int); cl=ref["cluster"]; paths=ref["img_path"]
    gc,age=clean_groups(paths,scheme,pf)
    print(f"== {s}: n={len(y)} frame agreement Xu vs AI-Face group {np.mean(gx==gc):.3f}")
    # group sizes under each labeling
    for lab,g in (("Xu",gx),("AIFace",gc)):
        names=SCHEMES[scheme]
        print("  ",lab,{names[k]:(int(((g==k)&(y==0)).sum()),int(((g==k)&(y==1)).sum()),len(set(cl[(g==k)&(y==0)])),len(set(cl[(g==k)&(y==1)]))) for k in range(len(names))})
    pt={}
    for nm,L in M.items():
        for lab,g in (("Xu",gx),("AIFace",gc)):
            ps=[metr(y,z[s],g,scheme,b,age) for b,z in L]
            pt[(nm,lab)]={k:float(np.nanmean([p[k] for p in ps])) for k in ps[0]}
    for k_,v in pt.items(): print("  ",k_,{a:round(b,2) for a,b in v.items()})
    # cluster bootstrap of D - Ding full under each labeling
    u,inv=np.unique(cl,return_inverse=True); mem=[np.where(inv==c)[0] for c in range(len(u))]
    rng=np.random.default_rng(2026); d={("Xu",k):[] for k in "i g r age".split()}; d.update({("AIFace",k):[] for k in "i g r age".split()})
    for _ in range(B):
        idx=np.concatenate([mem[c] for c in rng.integers(0,len(mem),len(mem))]); yy=y[idx]
        if len(np.unique(yy))<2: continue
        for lab,g in (("Xu",gx),("AIFace",gc)):
            a=[metr(yy,z[s][idx],g[idx],scheme,b,age[idx]) for b,z in M["CFES D"]]
            r=metr(yy,M["Ding full s7"][0][1][s][idx],g[idx],scheme,M["Ding full s7"][0][0],age[idx])
            for k in "i g r age".split(): d[(lab,k)].append(np.nanmean([x[k] for x in a])-r[k])
    ci={f"{lab}|{k}":[pt[("CFES D",lab)][k]-pt[("Ding full s7",lab)][k],float(np.nanpercentile(v,2.5)),float(np.nanpercentile(v,97.5))] for (lab,k),v in d.items()}
    for k,v in ci.items(): print("   D-Ding",k,[round(x,2) for x in v])
    res[s]={"points":{f"{a}|{b}":v for (a,b),v in pt.items()},"ci_D_minus_Ding":ci,"n_clusters":len(u)}
json.dump(res,open(os.path.dirname(os.path.abspath(__file__))+"/relabel_results.json","w"),indent=1)
