import pandas as pd, os, re, numpy as np
P="/home/sachin.chaudhary/models_external/robust_fairness_withNoisy_labels/dataset"
D="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/data"
R={0:'A',1:'W',2:'B',3:'O'}
pairs={"ffpp_train":(["ff++/train_real.csv","ff++/train_fake_spe.csv"],["ff++_ding/realtrain.csv","ff++_ding/faketrain.csv"]),
"ffpp_test":(["ff++/test_noise.csv"],["ff++/test.csv"]),"dfdc":(["dfdc/test_noise.csv"],["dfdc/test.csv"]),
"celebdf":(["celebdf/test_noise.csv"],["celebdf/test.csv"]),"dfd":(["dfd/test_noise.csv"],["dfd/test.csv"])}
vid=lambda p: re.sub(r'_\d+$','',re.sub(r'^\d+_','',os.path.basename(p)[:-4]))
rows=[]
for k,(pf,df_) in pairs.items():
    p=pd.concat([pd.read_csv(os.path.join(P,f)) for f in pf]); p['b']=p['Image Path'].map(os.path.basename)
    ndup=p.b.duplicated().sum(); conf=p.groupby('b')[['Ground Truth Gender','Ground Truth Race','Ground Truth Age']].nunique().max(axis=1).gt(1).sum()
    p=p.drop_duplicates('b')
    o=pd.concat([pd.read_csv(os.path.join(D,f)) for f in df_]); o['b']=o.img_path.map(os.path.basename)
    odup=o.b.duplicated().sum()
    m=o.merge(p,on='b',how='inner')
    m['g_our']=np.where(m.ismale==1,'M','F')
    def rr(r):
        for c,n in [('isasian','A'),('iswhite','W'),('isblack','B')]:
            if c in m.columns and r[c]==1: return n
        return 'O'
    m['r_our']=m.apply(rr,axis=1)
    m['g_cl']=np.where(m['Ground Truth Gender']==1,'M','F'); m['r_cl']=m['Ground Truth Race'].map(R)
    m['vid']=m.b.map(vid)
    ga=(m.g_our==m.g_cl).mean(); ra=(m.r_our==m.r_cl).mean(); ia=((m.g_our==m.g_cl)&(m.r_our==m.r_cl)).mean()
    # restrict race comparison to races both vocabularies have (celebdf/dfd lack Asian in noisy)
    print(f"== {k}: our rows {len(o)} (dup paths {odup}), purdue dup paths {ndup} (conflicting {conf}), matched {len(m)}")
    print(f"   frame agreement ours(noisy, Xu et al.) vs AI-Face 'cleaner': gender {ga:.3f} race {ra:.3f} intersection {ia:.3f}")
    for lab in [0,1]:
        mm=m[m.label==lab]
        print(f"    label={lab}: gender {(mm.g_our==mm.g_cl).mean():.3f} race {(mm.r_our==mm.r_cl).mean():.3f} inter {((mm.g_our==mm.g_cl)&(mm.r_our==mm.r_cl)).mean():.3f}")
    print("   gender confusion (rows ours, cols clean):\n", pd.crosstab(m.g_our,m.g_cl).to_string())
    print("   race confusion:\n", pd.crosstab(m.r_our,m.r_cl).to_string())
    # per-our-group: share of frames whose clean label differs
    m['gi']=m.g_our+'-'+m.r_our; m['gc']=m.g_cl+'-'+m.r_cl
    m['ok']=(m.gi==m.gc); pr=m.groupby('gi').agg(n=('ok','size'),precision_vs_clean=('ok','mean'),videos=('vid','nunique'))
    print("   per our-group precision vs clean:\n", pr.round(3).to_string())
    m[['b','label','g_our','r_our','g_cl','r_cl','Ground Truth Age','vid']].to_csv(f"/tmp/claude-156900507/-home-sachin-chaudhary/3e01ae21-517f-480b-a721-cb2115803d08/scratchpad/joined_{k}.csv",index=False)
