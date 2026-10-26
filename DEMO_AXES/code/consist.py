import pandas as pd, re, os
D="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/data"
sets={"ffpp_all":["ff++_ding/realtrain.csv","ff++_ding/faketrain.csv","ff++_ding/realval.csv","ff++_ding/fakeval.csv","ff++/test.csv"],"dfdc":["dfdc/test.csv"],"celebdf":["celebdf/test.csv"],"dfd":["dfd/test.csv"]}
for s,fs in sets.items():
    df=pd.concat([pd.read_csv(os.path.join(D,f)).assign(src=f) for f in fs])
    b=df.img_path.map(lambda p: re.sub(r'_\d+$','',re.sub(r'^\d+_','',os.path.basename(p)[:-4])))
    df['vid']=b
    n=df.groupby('vid').intersec_label.nunique()
    print(s,"videos",len(n),"videos with >1 intersec label:",(n>1).sum(), "frames in them:", df[df.vid.isin(n[n>1].index)].shape[0])
    if s=="ffpp_all":
        # real youtube_XXX vs fake Method_XXX_YYY: target = first id
        real=df[df.label==0].copy(); real['id']=real.vid.str.extract(r'youtube_(\d+)')[0]
        fake=df[df.label==1].copy(); fake['tgt']=fake.vid.str.extract(r'_(\d+)_(\d+)$')[0]; fake['srcid']=fake.vid.str.extract(r'_(\d+)_(\d+)$')[1]
        rl=real.groupby('id').intersec_label.agg(lambda x:x.mode()[0])
        fl=fake.groupby(['vid','tgt','srcid']).intersec_label.agg(lambda x:x.mode()[0]).reset_index()
        fl['rt']=fl.tgt.map(rl); fl['rs']=fl.srcid.map(rl)
        m=fl.dropna(subset=['rt'])
        print(" fake label == real(target a) label:",(m.intersec_label==m.rt).mean().round(3),"n",len(m)," == real(source b):",(fl.dropna(subset=['rs']).intersec_label==fl.dropna(subset=['rs']).rs).mean().round(3))
        print(" methods:", fake.vid.str.extract(r'^([A-Za-z]+)_')[0].value_counts().to_dict())
        print(" real ids",real.id.nunique())
    if s=="celebdf":
        print(df.vid.str.extract(r'^(Celeb-real|Celeb-synthesis|YouTube-real)')[0].value_counts().to_dict())
        df['ident']=df.vid.str.extract(r'(id\d+)')[0]
        print(" identities", df.ident.nunique(), "ident w/ >1 label:", (df.groupby('ident').intersec_label.nunique()>1).sum())
        print(df.vid.head(3).tolist(), df[df.label==1].vid.head(3).tolist())

print("---- per-method agreement FF++")
fs=sets["ffpp_all"]
df=pd.concat([pd.read_csv(os.path.join(D,f)) for f in fs])
df['vid']=df.img_path.map(lambda p: re.sub(r'_\d+$','',re.sub(r'^\d+_','',os.path.basename(p)[:-4])))
real=df[df.label==0].copy(); real['id']=real.vid.str.extract(r'youtube_(\d+)')[0]
rl=real.groupby('id').agg(il=('intersec_label','first'),m=('ismale','first'))
fake=df[df.label==1].drop_duplicates('vid').copy()
x=fake.vid.str.extract(r'^([A-Za-z]+)_(\d+)_(\d+)$'); fake['meth']=x[0]; fake['a']=x[1]; fake['b']=x[2]
for k in ['a','b']:
    fake['il_'+k]=fake[k].map(rl.il); fake['m_'+k]=fake[k].map(rl.m)
for meth,g in fake.groupby('meth'):
    ga=g.dropna(subset=['il_a']); gb=g.dropna(subset=['il_b'])
    print(meth,len(g),"agree(a) intersec %.3f gender %.3f | agree(b) intersec %.3f gender %.3f"%((ga.intersec_label==ga.il_a).mean(),(ga.ismale==ga.m_a).mean(),(gb.intersec_label==gb.il_b).mean(),(gb.ismale==gb.m_b).mean()))
# real youtube vs CelebDF identities
cd=pd.read_csv(os.path.join(D,"celebdf/test.csv")); cd['vid']=cd.img_path.map(lambda p: re.sub(r'_\d+$','',re.sub(r'^\d+_','',os.path.basename(p)[:-4])))
v=cd.drop_duplicates('vid').copy()
x=v.vid.str.extract(r'Celeb-synthesis_(id\d+)_(id\d+)_'); v['a']=x[0]; v['b']=x[1]
r=v[v.vid.str.startswith('Celeb-real')].copy(); r['a']=r.vid.str.extract(r'(id\d+)')[0]
rl=r.groupby('a').ismale.agg(lambda s: s.mode()[0])
f=v[v.vid.str.startswith('Celeb-synthesis')]
print("celebdf real ids",r.a.nunique(),"fake vids",len(f),"fake target ids",f.a.nunique(), "fake ismale counts",f.ismale.value_counts().to_dict())
print("real id gender", rl.value_counts().to_dict())
ff=f.dropna(subset=['a']).copy(); ff['ra']=ff.a.map(rl); ff['rb']=ff.b.map(rl)
print("fake gender==real target(a) gender:", (ff.dropna(subset=['ra']).ismale==ff.dropna(subset=['ra']).ra).mean(), len(ff.dropna(subset=['ra'])), " ==b:",(ff.dropna(subset=['rb']).ismale==ff.dropna(subset=['rb']).rb).mean())
