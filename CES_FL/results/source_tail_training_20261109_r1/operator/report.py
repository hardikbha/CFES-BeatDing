"""CPU-only aggregation of saved source-tail outcomes. No images, TEST/CAL or GPU calls."""
import argparse,csv,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
src=Path(a.source);root=Path(a.root);out=Path(a.out);torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('st',src/'CES_FL/proposals/source_tail_training_20261109_r1/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
    if not rows:return
    fields=sorted({k for r in rows for k in r})
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def cp(name):
    f=root/name
    if f.exists():(out/name.replace('/','_')).write_bytes(f.read_bytes())
for name in ['CPU_INPUT_BINDINGS.json','CUDA_BINDING.json','CUDA_CHECKS.json','REAL_TRAIN_PROFILE.json','ALLOCATION.json','PROCESS.json','COHORT_ALLOCATION_END.json',
             'COHORT_DONE.json','SOURCE_RECOVERY.json','qualify_profile.RECEIPT.json','prepare.RECEIPT.json','train_control.RECEIPT.json','train_treatment.RECEIPT.json',
             'export_control.RECEIPT.json','export_treatment.RECEIPT.json','budget_session/SESSION.json']:
    cp(name)
if not (root/'COHORT_ALLOCATION_END.json').exists():raise SystemExit('Cohort still active; no final report claimed')
statuses={};hist=[]
for arm in ['control','treatment']:
    f=root/arm/'RUN_VERIFIED.json'
    statuses[arm]=read(f)['status'] if f.exists() else 'BLOCKED_INPUT_OR_STAGE_FAILURE'
    if f.exists():write(out/(arm+'_RUN_VERIFIED.json'),read(f))
    if (root/arm/'history.json').exists():
        h=read(root/arm/'history.json');write(out/(arm+'_HISTORY.json'),h)
        for epoch in h:hist.append({'arm':arm,'epoch':epoch['epoch'],'steps':epoch['steps'],'feasible':epoch['feasible'],
            'auc':epoch['validation']['overall']['auc'],'fnr':epoch['validation']['overall']['fnr'],
            'joint_F_FPR':epoch['validation']['axes']['gender_race']['F_FPR'],**epoch['losses_step_mean']})
    export=root/(arm+'_student.pt.VERIFIED.json')
    statuses[arm+'_export']='PASS' if export.exists() and read(export)['status']=='PASS' else 'BLOCKED_OR_FAILED'
    if export.exists():write(out/(arm+'_EXPORT_VERIFIED.json'),read(export))
csvout('EPOCH_SUMMARY.csv',hist)
write(out/'ARM_STATUS.json',statuses)
cache=root/'cache'
if not (cache/'CACHE_VERIFIED.json').exists():
    write(out/'FINAL_STATUS.json',{'status':'BLOCKED_CACHE_OR_QUALIFICATION','arms':statuses,'scientific_config_changed':False,'no_automatic_extension':True})
    print(json.dumps({'status':'BLOCKED_CACHE_OR_QUALIFICATION','arms':statuses}));raise SystemExit(0)
meta=m.checked_cache(cache);write(out/'CACHE_VERIFIED.json',meta)
rows=read(cache/'val_rows_PRIVATE.json');train=read(cache/'train_rows_PRIVATE.json');y=np.array([r['label'] for r in rows]);model_data={}
model_data['teacher']={'z':np.load(cache/'val_teacher_logits.npy').astype(float),'threshold':meta['teacher_validation']['threshold_logit'],'report':meta['teacher_validation']}
all_metrics=[];group_rows=[];source_rows=[];exposure=[];params=[];source_risk=[]
teacher_ck=torch.load(Path(meta['teacher_run'])/'best.pt',map_location='cpu',weights_only=True)
teacher_head=torch.load(cache/'teacher_head.pt',map_location='cpu',weights_only=True)
for k in ['weight','bias']:assert torch.equal(teacher_head[k],teacher_ck['model']['classifier.'+k])
source_codes=np.load(cache/'real_sources.npy');tr_y=np.array([r['label'] for r in train])
reference=read(cache/'reference.json');scale=float(reference['scale']);thresholds=np.array(reference['thresholds']);targets=np.array(reference['targets'])
for arm in ['control','treatment']:
    f=root/arm/'validation_scores_PRIVATE.json'
    if not f.exists():continue
    r=read(f);v=read(root/arm/'RUN_VERIFIED.json');ck=torch.load(root/arm/'best.pt',map_location='cpu',weights_only=True)
    assert m.core.sha(root/arm/'best.pt')==v['checkpoint_sha256']==r['checkpoint_sha256']
    assert r['manifest_sha256']==meta['manifest_sha256']['val'] and r['split']=='validation' and len(r['rows'])==len(rows)
    assert [r0['image'] for r0 in r['rows']]==[r0['image'] for r0 in rows]
    z=np.array(r['logits'],float);report=m.core.metric_report(rows,z,r['threshold'],20);assert report==v['validation']
    model_data[arm]={'z':z,'threshold':r['threshold'],'report':report,'head':ck['head']}
    if (root/(arm+'_student.pt')).exists():
        student=torch.load(root/(arm+'_student.pt'),map_location='cpu',weights_only=True)
        encoder=[k for k in teacher_ck['model'] if k.startswith('encoder.')]
        assert all(torch.equal(teacher_ck['model'][k],student['model'][k]) for k in encoder)
        for k in ['weight','bias']:assert torch.equal(ck['head'][k],student['model']['classifier.'+k])
        params.append({'arm':arm,'encoder_tensors_bitwise_equal':len(encoder),'selected_head_equals_exported_head':True,'trainable_parameters':2049})
    for ep in read(root/arm/'history.json'):
        for s in [0,1]:
            for g in sorted({r['gender_race'] for r in train}):
                n=sum(source_codes[i]==s and r0['gender_race']==g for i,r0 in enumerate(train))
                exposure.append({'arm':arm,'epoch':ep['epoch'],'source':s,'group':g,'auxiliary_exposures':ep['auxiliary_source_joint_counts'].get(str(s)+':'+g,0),'unique_pool_rows':n})
write(out/'ENCODER_EXPORT_CHECKS.json',{'teacher_head_preserved_in_cache':True,'checks':params})
# Recorded source mapping imported only for metadata interpretation, never a trainer.
aud=Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-source-audit-results-20261109/CES_FL/results/ff2_source_stratified_audit_20261109_r1')
sys.path.insert(0,str(aud));import audit as audited;from audit_stats import connected_components,AUC
source=np.array([audited.source(r) for r in rows]);groups=sorted({r['gender_race'] for r in rows})
for name,d in model_data.items():
    for view,t in [('VAL10',d['threshold']),('fixed0.5',0.)]:
        report=m.core.metric_report(rows,d['z'],t,20)
        for axis,rr in report['axes'].items():
            all_metrics.append({'model':name,'view':view,'axis':axis,'threshold_logit':t,**report['overall'],
                **{k:rr.get(k) for k in ['F_FPR','F_DP','es_AUC','worst_group_auc','fpr_gap','fnr_gap']},'eligible_groups':len(rr['eligible_groups'])})
            for g,gg in rr['groups'].items():group_rows.append({'model':name,'view':view,'axis':axis,'group':g,'eligible':g in rr['eligible_groups'],**gg})
        for s in ['FFHQ','FairFace']:
            for g in ['ALL']+groups:
                ix=np.array([i for i,r in enumerate(rows) if r['label']==0 and source[i]==s and (g=='ALL' or r['gender_race']==g)],int)
                source_rows.append({'model':name,'view':view,'source':s,'group':g,'n_real':len(ix),'false_positives':int((d['z'][ix]>=t).sum()),'fpr':float((d['z'][ix]>=t).mean()) if len(ix) else None})
        for fam,rr in report.get('by_fake_family_vs_all_reals',{}).items():group_rows.append({'model':name,'view':view,'axis':'family_vs_all_reals','group':fam,**rr})
    # Full TRAIN descriptive source-risk monitoring; no new encoder pass.
    h=torch.nn.Linear(2048,1);h.load_state_dict(teacher_head if name=='teacher' else d['head']);features=np.load(cache/'train_features.npy',mmap_mode='r')
    if name=='teacher':ztr=np.load(cache/'train_teacher_logits.npy').astype(float)
    else:ztr=m.predict_head(h,features,torch.device('cpu')).astype(float)
    risk=0.
    for s in [0,1]:
        q=1/(1+np.exp(-np.clip((ztr[source_codes==s,None]-thresholds)/(scale*reference['temperature']),-80,80)))
        rates=q.mean(0);risk+=float(np.square(np.maximum(rates-targets,0)).sum())
        for k,rate in enumerate(rates):source_risk.append({'model':name,'source':s,'quantile':(.85,.90,.95)[k],'fixed_teacher_threshold':thresholds[k],
            'soft_exceedance':float(rate),'fixed_pooled_teacher_target':float(targets[k]),'n_train_real':int((source_codes==s).sum())})
    fake_guard=float(np.square(np.maximum((np.load(cache/'train_teacher_logits.npy')[tr_y==1]-ztr[tr_y==1])/scale,0)).mean())
    d['full_train_source_tail']=risk/6;d['full_train_fake_guard']=fake_guard
csvout('VALIDATION_METRICS.csv',all_metrics);csvout('VALIDATION_GROUP_FAMILY_METRICS.csv',group_rows);csvout('VALIDATION_SOURCE_REAL_FPR.csv',source_rows)
csvout('AUXILIARY_SOURCE_JOINT_EXPOSURE.csv',exposure);csvout('FULL_TRAIN_SOURCE_RISKS.csv',source_risk)
write(out/'FULL_TRAIN_OBJECTIVE_DIAGNOSTIC.json',{name:{'source_tail':d['full_train_source_tail'],'fake_guard':d['full_train_fake_guard'],'role':'in-sample TRAIN descriptive, not checkpoint selection'} for name,d in model_data.items()})
# Paired recorded-contributor uncertainty, audited2000/seed17/95% rule.
allcomp=connected_components(train+rows);_,comp=np.unique(allcomp[len(train):],return_inverse=True);nc=int(comp.max())+1
names=list(model_data);axes=model_data['teacher']['report']['axes'];cats=[np.arange(len(rows))]
axis_indices={}
for axis,rr in axes.items():
    axis_indices[axis]=[]
    for g in rr['eligible_groups']:
        axis_indices[axis].append(len(cats));cats.append(np.array([i for i,r in enumerate(rows) if r[axis]==g],int))
nr=np.zeros((nc,len(cats)));nf=nr.copy();fps=np.zeros((len(names),nc,len(cats)));fns=fps.copy();plans=[]
for mi,name in enumerate(names):
    z=model_data[name]['z'];t=model_data[name]['threshold'];plans.append(AUC(y,z,comp,np.arange(len(rows))))
    for j,ix in enumerate(cats):
        ri=ix[y[ix]==0];fi=ix[y[ix]==1]
        if mi==0:nr[:,j]=np.bincount(comp[ri],minlength=nc);nf[:,j]=np.bincount(comp[fi],minlength=nc)
        fps[mi,:,j]=np.bincount(comp[ri],weights=(z[ri]>=t),minlength=nc);fns[mi,:,j]=np.bincount(comp[fi],weights=(z[fi]<t),minlength=nc)
keys=['AUC','FPR','FNR']+['F_FPR/'+axis for axis in axes]
def calculate(w):
    a=w@nr;b=w@nf
    fp=(w@fps.transpose(1,0,2).reshape(nc,-1)).reshape(len(names),-1);fn=(w@fns.transpose(1,0,2).reshape(nc,-1)).reshape(len(names),-1)
    with np.errstate(invalid='ignore',divide='ignore'):fpr=fp/a;fnr=fn/b
    vals=[np.array([plan(w) for plan in plans]),fpr[:,0],fnr[:,0]]
    for axis,ii in axis_indices.items():
        qq=fpr[:,ii];v=np.abs(qq-fpr[:,0,None]).sum(1);v[~np.isfinite(qq).all(1)]=np.nan;vals.append(v)
    return np.stack(vals,1)
point=calculate(np.ones(nc));draws=[];rng=np.random.default_rng(17)
for _ in range(2000):draws.append(calculate(rng.multinomial(nc,np.full(nc,1/nc))))
draws=np.array(draws);intervals=[]
if 'control' in names and 'treatment' in names:
    i=names.index('control');j=names.index('treatment')
    for k,key in enumerate(keys):
        v=draws[:,j,k]-draws[:,i,k];v=v[np.isfinite(v)];ok=len(v)>=1900
        intervals.append({'contrast':'treatment-minus-control','metric':key,'estimate':point[j,k]-point[i,k],
            'ci95_low':float(np.quantile(v,.025)) if ok else None,'ci95_high':float(np.quantile(v,.975)) if ok else None,
            'valid_replicates':len(v),'required_replicates':1900,'status':'EXPLORATORY_INTERVAL' if ok else 'INCONCLUSIVE_COMPONENT_SUPPORT'})
csvout('PAIRED_CONTRIBUTOR_INTERVALS.csv',intervals)
write(out/'UNCERTAINTY_SCOPE.json',{'components':nc,'bootstrap_replicates':2000,'analysis_seed':17,'minimum_valid':1900,'thresholds_refit':False,
    'unit':'Whole recorded source-master/contributor connected components, TRAIN+SELECT graph projected onto SELECT','multiple_axes':'unadjusted exploratory; shared encoder/selected-checkpoint conditioning','biological_identity_separation_verified':False})
write(out/'FINAL_STATUS.json',{'status':'COHORT_REPORTED_NO_AUTOMATIC_EXTENSION','arms':statuses,'source_hashes':m.hashes(),
    'teacher_head_preserved_at_initialization':True,'encoder_frozen':True,'encoder_cache_shared':True,'scientific_config_changed':False,
    'test_cal_read':False,'no_source_standardized_race_joint_estimate_manufactured':True})
print(json.dumps({'status':'REPORTED','arms':statuses,'models_with_selected_scores':names,'components':nc,'paired_intervals':intervals},indent=2))
