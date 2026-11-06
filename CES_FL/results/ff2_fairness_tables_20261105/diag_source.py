# Aggregate-only diagnostic: real-image FPR by source dataset x race on FF2 VAL, at each model's frozen VAL-real 10% threshold.
import json, numpy as np, collections
from pathlib import Path
BASE=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_baseline_suite_s17_20261103T110324Z')
REPLAY=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_intersection_reanalysis_20261105/gpu_shared_replay_retry1/ff2_age_skin_val_replay_20261105_gpu02_1793830972/scores')
MODELS={'BCE':BASE/'bce/training/validation_scores_PRIVATE.json','DAG-FDD':BASE/'dag/training/validation_scores_PRIVATE.json',
 'DAW-FDD':BASE/'daw/training/validation_scores_PRIVATE.json','CES-FL P':REPLAY/'CES_P/scores_PRIVATE.json',
 'CES-FL A':REPLAY/'CES_A/scores_PRIVATE.json','CES-FL B':REPLAY/'CES_B/scores_PRIVATE.json',
 'Ding corrected GDA':Path('/home/sachin.chaudhary/ces_fl_runs/ff2_intersection_reanalysis_20261105/aggregates_v1/ding_corrected_gda_scores_PRIVATE.json'),
 'L_A (plain-head BCE)':Path('/home/sachin.chaudhary/ces_fl_runs/ff2_lh_ps_20261106_r1/L_A/validation_scores_PRIVATE.json')}
VAL_SHA='aeda051259d9e26b07c2810152b601ae115ce7c9e10523f99ea07bbc9f92afa8'
out={}
for m,p in MODELS.items():
    d=json.load(open(p)); assert d['manifest_sha256']==VAL_SHA and len(d['rows'])==31107
    z=np.asarray(d['logits'],float); t=float(d.get('threshold',d.get('threshold_logit')))
    cnt=collections.defaultdict(lambda:[0,0])
    for r,s in zip(d['rows'],z):
        if int(r['label'])!=0: continue
        src=r['source_ids'].split(':')[0]
        for k in [(src,'ALL'),(src,r['race']),(src,r['gender_race']),('ALL',r['race']),(src,'age'+r['age_bin'])]:
            cnt[k][0]+=1; cnt[k][1]+=int(s>=t)
    out[m]={f'{a}|{b}':[n,fp] for (a,b),(n,fp) in cnt.items()}
json.dump(out,open('diag_source_fpr.json','w'),indent=1)
keys=sorted(out['BCE'])
print('cell'.ljust(30),'n'.rjust(5),*[m[:9].rjust(10) for m in MODELS])
for k in keys:
    n=out['BCE'][k][0]
    print(k.ljust(30),str(n).rjust(5),*[f"{100*out[m][k][1]/out[m][k][0]:10.2f}" for m in MODELS])
