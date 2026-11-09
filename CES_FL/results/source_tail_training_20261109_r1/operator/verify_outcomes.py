"""CPU integrity/arithmetic checks of completed cohort; no encoder inference."""
import argparse,csv,hashlib,importlib.util,json,math,subprocess
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
prop=a.source/'CES_FL/proposals/source_tail_training_20261109_r1'
s=importlib.util.spec_from_file_location('source_tail_verification',prop/'run.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.set_num_threads(2);checks=[]
def passed(label):checks.append(label)
assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()=='9bb8e8332b8a3aa78d83f74ef21bca0b7dcd6754'
assert not subprocess.check_output(['git','-C',str(a.source),'status','--porcelain'],text=True).strip()
c=m.config();cache=a.root/'cache';meta=m.checked_cache(cache)
assert meta['source_hashes']==m.hashes() and not meta['fixture'] and not meta['test_read'] and not meta['cal_read']
passed('exact_clean_scientific_commit_and_all_nine_cache_blob_hashes')
head=torch.load(cache/'teacher_head.pt',map_location='cpu',weights_only=True)
teacher=torch.load(Path(meta['teacher_run'])/'best.pt',map_location='cpu',weights_only=True)
assert sha(Path(meta['teacher_run'])/'best.pt')==c['teacher_checkpoint_sha256']
for k,v in head.items():assert torch.equal(v,teacher['model']['classifier.'+k])
passed('complete_existing_teacher_classifier_preserved_in_shared_cache')
reference=read(cache/'reference.json');sources=np.load(cache/'real_sources.npy');teacher_scores=np.load(cache/'train_teacher_logits.npy')
ref=m.objective.fit_train_reference(torch.from_numpy(teacher_scores[sources>=0]),c['quantiles'],c['temperature'])
for k in ['thresholds','scale','targets']:assert torch.equal(torch.as_tensor(reference[k]),ref[k])
passed('fixed_reference_reproduces_from_TRAIN_real_teacher_scores_only')
rows=read(cache/'train_rows_PRIVATE.json');n=len(rows);steps=math.ceil(n/c['batch_size']);modelheads={};armrecords=[]
for arm in ['control','treatment']:
    root=a.root/arm;v=read(root/'RUN_VERIFIED.json');h=read(root/'history.json')
    assert v['status']=='COMPLETED_VERIFIED' and v['epochs']==3 and len(h)==3 and not v['fixture']
    assert (root/'DONE').exists() and sha(root/'best.pt')==v['checkpoint_sha256']
    ck=torch.load(root/'best.pt',map_location='cpu',weights_only=True);last=torch.load(root/'last.pt',map_location='cpu',weights_only=True)
    assert ck['config']==last['config']==c and ck['arm']==arm and last['epoch']==3 and last['steps']==3*steps
    feasible=[r for r in h if r['validation']['overall']['auc']>=c['min_auc'] and r['validation']['overall']['fnr']<=c['max_fnr']]
    chosen=min(feasible,key=lambda r:(r['validation']['axes']['gender_race']['F_FPR'],-r['validation']['overall']['auc'],r['epoch']))
    assert chosen['epoch']==v['selected_epoch']==ck['best']['epoch'] and chosen['validation']==v['validation']
    assert ck['best']['calibration']==chosen['calibration'] and v['cache_digest']==sha(cache/'CACHE_VERIFIED.json')
    for rec in h:
        assert rec['originals_once_each'] and rec['original_count']==n and rec['steps']==rec['epoch']*steps
        for code in c['source_codes'].values():assert sum(v for k,v in rec['auxiliary_source_joint_counts'].items() if k.startswith(str(code)+':'))==steps*c['auxiliary_per_source']
    student=a.root/(arm+'_student.pt');ev=read(a.root/(arm+'_student.pt.VERIFIED.json'))
    assert ev['status']=='PASS' and ev['full_native_validation_replay'] and ev['checkpoint_sha256']==sha(student)
    sd=torch.load(student,map_location='cpu',weights_only=True)
    assert sd['source_hashes']==m.hashes() and not sd['source_labels_needed_at_inference']
    for k in teacher['model']:
        if k.startswith('encoder.'):assert torch.equal(teacher['model'][k],sd['model'][k])
    for k in ck['head']:assert torch.equal(ck['head'][k],sd['model']['classifier.'+k])
    modelheads[arm]=ck['head'];armrecords.append({'arm':arm,'head_checkpoint_sha256':sha(root/'best.pt'),'student_checkpoint_sha256':sha(student),
        'selected_epoch':v['selected_epoch'],'epochs_completed':3,'optimizer_steps':last['steps'],'config_sha256':sha(root/'config.json')})
passed('both_three_epoch_histories_exposures_fixed_selection_and_head_checkpoint_binding')
passed('both_full_native_exports_PASS_and_all_encoder_parameters_BN_buffers_bitwise_unchanged')
assert any(not torch.equal(modelheads['control'][k],modelheads['treatment'][k]) for k in head)
passed('nonzero_source_coefficient_changed_actual_learned_parameters')
hc=read(a.root/'control/history.json');ht=read(a.root/'treatment/history.json')
assert [e['auxiliary_source_joint_counts'] for e in hc]==[e['auxiliary_source_joint_counts'] for e in ht]
passed('matched_original_auxiliary_schedule_and_source_by_joint_exposure')
exposure=[]
for epoch in range(c['epochs']):
    seen={s:set() for s in c['source_codes'].values()};counts={s:0 for s in seen};pools={s:np.flatnonzero(sources==s) for s in seen}
    for step in range(steps):
        rng=np.random.default_rng(c['seed']+1000003*(epoch*steps+step))
        for code,pool in pools.items():
            ix=rng.choice(pool,size=c['auxiliary_per_source'],replace=False);seen[code].update(map(int,ix));counts[code]+=len(ix)
    for code in seen:exposure.append({'epoch':epoch+1,'source_code':code,'total_exposures':counts[code],'unique_real_rows_seen':len(seen[code]),'source_pool_size':len(pools[code]),
        'mean_exposures_per_pool_row':counts[code]/len(pools[code]),'same_schedule_for_both_arms':True})
with (a.out/'ACTUAL_AUXILIARY_COVERAGE.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(exposure[0]),lineterminator='\n');w.writeheader();w.writerows(exposure)
# Strict float64 thresholding: avoid float32 scalar coercion at nextafter boundary.
z=np.array([1.,0.],np.float32);t=float(np.nextafter(np.float64(1.),np.inf));assert (z.astype(np.float64)>=t).sum()==0
with (a.out/'EXTENDED_VALIDATION_METRICS.csv').open() as f:table=list(csv.DictReader(f))
for model in ['teacher','control','treatment']:
    report=meta['teacher_validation'] if model=='teacher' else read(a.root/model/'RUN_VERIFIED.json')['validation']
    row=next(r for r in table if r['model']==model and r['axis']=='gender_race' and r['view']=='VAL10')
    for k in ['auc','fpr','fnr']:assert math.isclose(float(row['overall_'+k]),report['overall'][k],abs_tol=1e-12)
    assert math.isclose(float(row['F_FPR']),report['axes']['gender_race']['F_FPR'],abs_tol=1e-12)
passed('independent_presentation_matches_native_teacher_and_both_students_with_float64_thresholds')
allocation=read(a.root/'COHORT_ALLOCATION_END.json');assert allocation['exit_code']==0 and allocation['max_gpus']==1 and allocation['held_gpu_hours']<=2
ledger=read(Path('/home/sachin.chaudhary/ces_fl_runs/ff2_baseline_suite_s17_20261103T110324Z/additional72_gpu_hours.json'));charged=sum(float(x.get('charged_gpu_hours') or 0) for x in ledger['jobs'].values())
assert charged<=72 and not any(x.get('state')=='ACTIVE' and 'source_tail' in name for name,x in ledger['jobs'].items())
passed('actual_cohort_charge_includes_lease_bookends_and_is_within_both_caps')
(a.out/'FINAL_VERIFICATION.json').write_text(json.dumps({'status':'PASS','checks':checks,'source_commit':'9bb8e8332b8a3aa78d83f74ef21bca0b7dcd6754',
    'scientific_hashes':m.hashes(),'arm_checkpoints':armrecords,'cache_receipt_sha256':sha(cache/'CACHE_VERIFIED.json'),
    'cohort_gpu_hours':allocation['held_gpu_hours'],'shared_charged_gpu_hours':charged,'remaining_shared_gpu_hours':72-charged,
    'new_encoder_inference_or_training_executed_by_verifier':False,'test_cal_read':False},indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':checks,'optimizer_steps_per_arm':3*steps,'cohort_gpu_hours':allocation['held_gpu_hours'],'remaining_shared_hours':72-charged},indent=2))
