import pandas as pd, os, re
A="/home/sachin.chaudhary/ICLR_2027/datasets_external/AI-Face/annotations"
D="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/data"
af=pd.concat([pd.read_csv(f"{A}/train_data.csv").assign(afsplit='train'),pd.read_csv(f"{A}/test_data.csv").assign(afsplit='test')])
af['ds']=af['Image Path'].str.extract(r'/deepfakes/([^/]+)/')[0]
print("AI-Face rows by deepfake ds:",af.ds.value_counts().to_dict())
print("Skin Tone Group values",af['Skin Tone Group'].value_counts().sort_index().to_dict(),"Age",af['Predicted Age'].value_counts().sort_index().to_dict())
print(af.groupby('Skin Tone Group')['Skin Tone'].agg(['min','max']))
af=af[af.ds.notna()].copy(); af['b']=af['Image Path'].map(os.path.basename)
sets={"ffpp_train":["ff++_ding/realtrain.csv","ff++_ding/faketrain.csv"],"ffpp_val":["ff++_ding/realval.csv","ff++_ding/fakeval.csv"],"ffpp_test":["ff++/test.csv"],"dfdc":["dfdc/test.csv"],"celebdf":["celebdf/test.csv"],"dfd":["dfd/test.csv"]}
dsmap={"ffpp_train":"ff++","ffpp_val":"ff++","ffpp_test":"ff++","dfdc":"dfdc","celebdf":"celebdf","dfd":"dfd"}
vid=lambda p: re.sub(r'_\d+$','',re.sub(r'^\d+_','',os.path.basename(p)[:-4]))
for s,fs in sets.items():
    o=pd.concat([pd.read_csv(os.path.join(D,f)) for f in fs]); o['b']=o.img_path.map(os.path.basename); o['vid']=o.img_path.map(vid)
    a=af[af.ds==dsmap[s]].drop_duplicates('b')
    m=o.merge(a[['b','Skin Tone','Skin Tone Group','Predicted Age','Predicted Gender','afsplit']],on='b',how='left')
    hit=m['Skin Tone'].notna()
    print(f"== {s}: rows {len(o)}, with AI-Face skin/age {hit.sum()} ({hit.mean():.1%}); videos covered {m[hit].vid.nunique()}/{m.vid.nunique()}; AI-Face split of matches {m[hit].afsplit.value_counts().to_dict()}")
    mm=m[hit]
    if len(mm):
        print("  skin tone group x label frames:",mm.groupby(['Skin Tone Group','label']).size().unstack(fill_value=0).to_dict())
        print("  skin tone group x label videos:",mm.groupby(['Skin Tone Group','label']).vid.nunique().unstack(fill_value=0).to_dict())
        print("  MST hist:",mm['Skin Tone'].value_counts().sort_index().to_dict())
        print("  age x label videos:",mm.groupby(['Predicted Age','label']).vid.nunique().unstack(fill_value=0).to_dict())
        # skin tone by Xu race
        def rr(r):
            for c,n in [('isasian','A'),('iswhite','W'),('isblack','B')]:
                if c in mm.columns and r[c]==1: return n
            return 'O'
        mm=mm.assign(race=mm.apply(rr,axis=1))
        print("  Xu race x skin group (frames):\n",pd.crosstab(mm.race,mm['Skin Tone Group']).to_string())
        mm[['b','label','vid','race','ismale','Skin Tone','Skin Tone Group','Predicted Age','Predicted Gender']].to_csv(f"/tmp/claude-156900507/-home-sachin-chaudhary/3e01ae21-517f-480b-a721-cb2115803d08/scratchpad/aiface_{s}.csv",index=False)
