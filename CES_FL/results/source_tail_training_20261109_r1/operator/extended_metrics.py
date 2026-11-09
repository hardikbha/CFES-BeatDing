"""CPU metrics from saved SELECT logits only; no model execution or TEST/CAL."""
import argparse,csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def read(p):return json.loads(p.read_text())
def writecsv(name,values):
    with (a.out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in values for k in r}),lineterminator='\n');w.writeheader();w.writerows(values)
def coarse(r):
    if r['gender'] not in ('Female','Male') or r['race'] not in ('Asian','Black','Others','White') or r['age_bin'] not in ('<30','30-44','45-59','60+'):return 'unknown'
    return r['gender']+'|'+('Youth' if r['age_bin']=='<30' else 'Adult')+'|'+('White' if r['race']=='White' else 'NonWhite')
def ageskin(r):return r['age_bin']+'|skin'+r['skin_bin'] if r['age_bin'] in ('<30','30-44','45-59','60+') and r['skin_bin'] in ('0','1','2','3') else 'unknown'
def source(r):
    keys=set(r['source_ids'].split(';'));labels=set()
    for k in keys:
        if k.startswith('ffhq:'):labels.add('FFHQ')
        elif k.startswith('img:images/real/C_fairface/'):labels.add('FairFace')
        else:labels.add('unknown')
    return next(iter(labels)) if len(labels)==1 else 'mixed_reference'
rows=read(a.root/'cache/val_rows_PRIVATE.json');teacher=read(a.root/'cache/CACHE_VERIFIED.json')
models={'teacher':(np.load(a.root/'cache/val_teacher_logits.npy').astype(np.float64),teacher['teacher_validation']['threshold_logit'])}
for arm in ('control','treatment'):
    d=read(a.root/arm/'validation_scores_PRIVATE.json');assert d['split']=='validation' and d['manifest_sha256']==teacher['manifest_sha256']['val']
    assert len(d['rows'])==len(rows) and [r['image'] for r in rows]==[r['image'] for r in d['rows']]
    models[arm]=(np.array(d['logits']),d['threshold'])
y=np.array([r['label'] for r in rows]);src=np.array([source(r) for r in rows]);axes={ax:[r[ax] for r in rows] for ax in ['gender','race','gender_race','age_bin','skin_bin','ids8']}
axes['age_x_skin16']=[ageskin(r) for r in rows];axes['coarse_gender_age_race8']=[coarse(r) for r in rows]
summary=[];group_rows=[];sources=[]
def stats(ix,z,pred):
    yy=y[ix];pp=pred[ix];r=yy==0;f=yy==1
    return {'n':len(ix),'n_real':int(r.sum()),'n_fake':int(f.sum()),'auc':float(roc_auc_score(yy,z[ix])) if r.any() and f.any() else None,
        'fpr':float(pp[r].mean()) if r.any() else None,'fnr':float((~pp[f]).mean()) if f.any() else None,
        'positive_rate':float(pp.mean()) if len(ix) else None,'accuracy':float((pp==yy).mean()) if len(ix) else None}
for model,(z,thr) in models.items():
    z=np.asarray(z,dtype=np.float64) # Preserve nextafter-calibrated float64 threshold decisions.
    assert z.shape==y.shape and np.isfinite(z).all()
    for view,t in [('VAL10',thr),('fixed0.5',0.)]:
        pred=z>=t;overall=stats(np.arange(len(rows)),z,pred)
        for axis,labels in axes.items():
            rr=[]
            for g in sorted(set(labels)):
                ix=np.flatnonzero(np.array(labels)==g);d=stats(ix,z,pred);ok=g!='unknown' and d['n_real']>=20 and d['n_fake']>=20
                group_rows.append({'model':model,'view':view,'axis':axis,'group':g,'eligible':ok,**d})
                if ok:rr.append(d)
            assert len(rr)>=2
            gap=lambda key:max(r[key] for r in rr)-min(r[key] for r in rr)
            summary.append({'model':model,'view':view,'axis':axis,'threshold_logit':t,'min_support_per_class':20,'eligible_groups':len(rr),
                **{'overall_'+k:v for k,v in overall.items()},'F_FPR':sum(abs(d['fpr']-overall['fpr']) for d in rr),
                'F_DP':gap('positive_rate'),'es_AUC':overall['auc']/(1+sum(abs(d['auc']-overall['auc']) for d in rr)),
                'FPR_gap':gap('fpr'),'FNR_gap':gap('fnr'),'F_MEO_max':max(gap('fpr'),gap('fnr')),
                'F_MEO_mean':.5*(gap('fpr')+gap('fnr')),'F_OAE':gap('accuracy'),'worst_group_auc':min(d['auc'] for d in rr),
                'annotation_caveat':'age/skin/ids8 are inherited estimates; unknown retained in native totals'})
        for s in ('FFHQ','FairFace'):
            for axis in ('gender','race','gender_race'):
                for g in sorted(set(axes[axis])):
                    ix=np.flatnonzero((src==s)&(np.array(axes[axis])==g));d=stats(ix,z,pred)
                    sources.append({'model':model,'view':view,'source':s,'axis':axis,'group':g,
                        'source_role':'contributing-reference acquisition for fakes, documented acquisition for reals','source_standardized_estimate':False,**d})
        for family in sorted({r['family'] for r in rows if r['label']==1}):
            for s in ('FFHQ','FairFace'):
                mask=(src==s)&((y==0)|np.array([r['label']==1 and r['family']==family for r in rows]))
                sources.append({'model':model,'view':view,'source':s,'axis':'family_vs_source_reals','group':family,
                    'source_role':'contributing-reference acquisition for fakes','source_standardized_estimate':False,**stats(np.flatnonzero(mask),z,pred)})
writecsv('EXTENDED_VALIDATION_METRICS.csv',summary);writecsv('EXTENDED_GROUP_METRICS.csv',group_rows);writecsv('SOURCE_PARENT_JOINT_AND_FAMILY_UTILITY.csv',sources)
# Match independent arithmetic against the native report wherever definitions coincide.
for arm in ('teacher','control','treatment'):
    native=teacher['teacher_validation'] if arm=='teacher' else read(a.root/arm/'RUN_VERIFIED.json')['validation']
    ref=next(r for r in summary if r['model']==arm and r['view']=='VAL10' and r['axis']=='gender_race')
    for k in ['auc','fpr','fnr']:assert math.isclose(ref['overall_'+k],native['overall'][k],abs_tol=1e-12)
    for ax,v in native['axes'].items():
        r=next(x for x in summary if x['model']==arm and x['view']=='VAL10' and x['axis']==ax)
        for k in ('F_FPR','F_DP','es_AUC','worst_group_auc'):assert math.isclose(r[k],v[k],abs_tol=1e-12),(arm,ax,k)
(a.out/'EXTENDED_METRIC_CHECKS.json').write_text(json.dumps({'status':'PASS','summary_rows':len(summary),'group_rows':len(group_rows),
    'source_parent_joint_family_rows':len(sources),'native_overlapping_axis_summaries_independently_match':True,
    'new_model_inference':False,'thresholds_refit':False,'interval_role':'Point-only for supplementary intersections; paired native-axis intervals are separate.',
    'units':'Rates/AUC/summed disparity fractions; times100 in narrative tables.'},indent=2)+'\n')
print(json.dumps({'status':'PASS','summary_rows':len(summary),'group_rows':len(group_rows),'source_breakdowns':len(sources)}))
