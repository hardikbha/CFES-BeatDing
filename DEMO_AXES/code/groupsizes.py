import pandas as pd, re, os
D="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/data"
sets={"ffpp_train":["ff++_ding/realtrain.csv","ff++_ding/faketrain.csv"],"ffpp_val":["ff++_ding/realval.csv","ff++_ding/fakeval.csv"],
"ffpp_test":["ff++/test.csv"],"dfdc":["dfdc/test.csv"],"celebdf":["celebdf/test.csv"],"dfd":["dfd/test.csv"]}
def vid(p,ds):
    b=os.path.basename(p)[:-4]
    b=re.sub(r'^\d+_','',b)  # leading index
    b=re.sub(r'_\d+$','',b)  # frame
    if ds=="dfd":
        # actor ids: e.g. 10_19__exit_phone_room__BVVPN5D5 (fake: a_b__scene__hash) or 10__scene (real)
        m=re.match(r'^(\d+)(?:_(\d+))?__',b)
        if m: return m.group(1)
    return b
out=[]
for s,fs in sets.items():
    df=pd.concat([pd.read_csv(os.path.join(D,f)) for f in fs])
    ds="dfd" if s=="dfd" else s
    df['vid']=[vid(p,ds) for p in df.img_path]
    df['g']=('M' if True else '')
    df['gender']=df.ismale.map({1:'M',-1:'F'})
    def race(r):
        for c,n in [('isasian','A'),('iswhite','W'),('isblack','B')]:
            if c in df.columns and r[c]==1: return n
        return 'O'
    df['race']=df.apply(race,axis=1)
    print("==",s,len(df), "intersec map:",df.groupby('intersec_label')[['gender','race']].agg(lambda x:x.mode()[0]).to_dict('index'))
    t=df.groupby(['gender','race','label']).agg(frames=('img_path','size'),videos=('vid','nunique')).unstack('label').fillna(0).astype(int)
    print(t)
    # marginals
    for ax in ['gender','race']:
        print(df.groupby([ax,'label']).agg(frames=('img_path','size'),videos=('vid','nunique')).unstack('label').fillna(0).astype(int))
