"""Read-only TRAIN/SELECT frozen-score audit. No model execution, images, TEST/CAL or CUDA queries."""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import posixpath
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from audit_stats import connected_components, run_statistics

HERE=Path(__file__).resolve().parent
DOC='76dcb54e240dcf56475dbf587836cdadcdad5aab'
SCI='e7fe0ee03e6cfacca81c558bbf632d2f72e80bc0'
TRAIN=Path('/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/train.csv')
SELECT=TRAIN.with_name('val.csv')
RUNS=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_rfa_20261107')
STUDIES=[('RF-S_s17','S'),('RF-S_s7','S_s7'),('RF-S_s27','S_s27'),
         ('RF-SJg_s17','SJ_gentle'),('RF-SJg_s7','SJ_gentle_s7'),('RF-SJg_s27','SJ_gentle_s27')]
HISTORY={'5ee7c8b92c0b5def8219d100439c9ba8cb4541e604af79fe41b7dff7a281b8c8':'0a135a03582f4738868b19ea0b3d81a0b6625658',
         '2d99295cc0b8190a571695b581ef5ba73e7b5fc18994461bdf1a21c1d83089d1':'ac6cba4671000f731903ac0b3946f1847678e3e5'}
CORE='8c25ba8cb844b059adbbce75b57785a156333182afa2d5d576985997f2d36733'
ROOT=Path('/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009')
DOCFILES=[Path('/home/sachin.chaudhary/ces_pilot_seed17/build_manifest.py'),
          Path('/home/sachin.chaudhary/ces_fl_pipeline/cesfl/prepare.py'),
          ROOT/'release_ff2_public/build_public.py',
          ROOT/'agents/T_scaleup/real_tiers/code/materialize.py',
          ROOT/'agents/T_scaleup/real_tiers/code/freeze_manifest.py']


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,allow_nan=False).encode()).hexdigest()
def codehashes():return {p.name:sha(p) for p in [HERE/'audit.py',HERE/'audit_stats.py']}
def tokens(s):return sorted({x.strip() for x in str(s).split(';') if x.strip()})
def git(root,*a):return subprocess.check_output(['git','-C',str(root),*a],text=True).strip()
def rows(path):
    if Path(path) not in (TRAIN,SELECT):raise ValueError('Only explicitly allowed TRAIN/SELECT manifests may be read')
    with Path(path).open(newline='') as f:out=list(csv.DictReader(f))
    seen=set()
    for r in out:
        r['image']=posixpath.normpath(r['image']);r['label']=int(r['label'])
        if r['label'] not in (0,1) or r['image'] in seen:raise ValueError('Invalid label or duplicate image in manifest')
        seen.add(r['image'])
        for k in ('identity_ids','source_ids'):
            if not tokens(r[k]):raise ValueError('Missing recorded component links')
            r[k]=';'.join(tokens(r[k]))
        for k in ('gender','race','gender_race','age_bin','skin_bin','family','method','ids8'):
            r[k]=str(r.get(k) or 'unknown').strip() or 'unknown'
        if r['gender_race']!=r['gender']+'-'+r['race']:raise ValueError('Joint/parent demographic mismatch')
    return out


def source(row):
    """Recorded canonical source keys, not appearance or convenience cluster IDs.

    build_manifest.real_key explicitly defines ffhq keys from src_dataset=FFHQ.
    For other keys, release TIERNAME and materialize.TD define C_fairface.
    Fake classification is reference-source provenance, not pixel acquisition.
    """
    sources=set()
    for t in tokens(row['source_ids']):
        if re.fullmatch(r'ffhq:\d{5}',t):sources.add('FFHQ')
        elif t.startswith('img:images/real/C_fairface/'):sources.add('FairFace')
        else:sources.add('UNRESOLVED')
    if len(sources)!=1:return 'MIXED_REFERENCE'
    s=next(iter(sources))
    if row['label']==0:
        if len(tokens(row['source_ids']))!=1:return 'UNRESOLVED'
        if s=='FFHQ':
            if '/real/A_ffhq/' not in row['image'] or Path(row['image']).stem.zfill(5)!=tokens(row['source_ids'])[0][5:]:return 'UNRESOLVED'
        elif s=='FairFace':
            rel=tokens(row['source_ids'])[0][4:]
            expected='/home/sachin.chaudhary/ICLR_2027/fairforge_scaleup/real/'+rel[len('images/real/'):]
            if expected!=row['image']:return 'UNRESOLVED'
    return s


def registry(paths):return {str(p):sha(p) for p in paths}
def census(data):
    out=[]
    for axis in ('gender','race','gender_race'):
        for g in sorted({r[axis] for r in data}):
            for s in ('FFHQ','FairFace'):
                rr=[r for r in data if r['label']==0 and r[axis]==g and source(r)==s]
                out.append({'axis':axis,'group':g,'source':s,'n_real':len(rr),
                            'n_recorded_masters':len({r['source_ids'] for r in rr})})
    return out


def freeze(args):
    if (HERE/'FREEZE.json').exists():raise FileExistsError('Preserve existing freeze; use versioned suffix')
    if (args.private/'SCORES_OPENED.json').exists():raise ValueError('Private outcomes already opened for this diagnostic')
    assert git(args.source,'rev-parse','HEAD')==DOC
    assert subprocess.run(['git','-C',str(args.source),'symbolic-ref','-q','HEAD'],capture_output=True).returncode!=0
    changed=git(args.source,'diff','--name-only',SCI,DOC).splitlines()
    assert len(changed)==3 and all(p.startswith('CES_FL/reviews/ff2_fairness_gap_20261109/') for p in changed)
    train=rows(TRAIN);counts=collections.Counter(source(r) for r in train if r['label']==0)
    if set(counts)!={'FFHQ','FairFace'}:raise ValueError('TRAIN acquisition provenance incomplete; cannot freeze full source estimand')
    groups={ax:sorted({r[ax] for r in train}) for ax in ('gender','race','gender_race')}
    data={'status':'FROZEN_BEFORE_PRIVATE_SCORE_OUTCOMES','frozen_utc':now(),'documentation_commit':DOC,
          'scientific_review_base':SCI,'detached_source_checkout':str(args.source),'helper_sha256':codehashes(),
          'sources':sorted(counts),'train_real_source_counts':dict(counts),
          'source_weights':{s:n/sum(counts.values()) for s,n in counts.items()},'groups':groups,
          'source_definition':'Documented canonical real acquisition dataset only: FFHQ or FairFace; per-image processing profiles are not inferred.',
          'source_mapping_evidence':'build_manifest.real_key, release TIERNAME A=FFHQ/C=FairFace, materialize.TD; real source-key/own-image exact metadata match required.',
          'train_real_rows':sum(counts.values()),'train_rows':len(train),'train_source_group_support':census(train),
          'inputs_sha256':registry([TRAIN,*DOCFILES]),
          'bootstrap':{'replicates':2000,'seed':17,'required_valid_fraction':.95,
            'unit':'Connected components of ALL recorded identity_ids/source_ids/image links; project TRAIN+SELECT graph onto SELECT.',
            'resampling':'Globally resample whole SELECT-bearing components with replacement; acquisition is an analytic stratum, never a two-dataset bootstrap unit. Cross-source components remain intact.',
            'pairing':'Same component multiplicities for every seed/model, native/standardized view and paired contrast.',
            'priors':'TRAIN source weights fixed; SELECT real-group proportions computed identically in both views, re-estimated together within each resample.',
            'intervals':'Percentile95%; only RF-S seed17 gender×race native-minus-standardized is primary. All other intervals unadjusted/exploratory; no multiplicity-adjusted claims.',
            'auc_intervals':'Native overall, gender×race and all-source fake-family AUC; other source/proxy AUC point estimates only.',
            'missing_support':'No imputation, cell dropping or weight renormalization. Undefined required cells block full estimand. Original group eligibility and thresholds fixed.'},
          'primary':'RF-S seed17: D_native minus D_standardized on gender×race',
          'secondary':'Repeat native-minus-standardized for both methods/all seeds; paired SJg-minus-S native and standardized, source residual gaps and native utility.',
          'metadata_limits':'Acquisition inferred only from documented identifiers; independent biological identity and individual crop/encoding/master pixel histories are not certified by these manifests.',
          'private_scores_read':False,'test_or_cal_accessed':False,'new_predictions_generated':False}
    write(HERE/'FREEZE.json',data)
    print(json.dumps({'stage':'FREEZE','status':'PASS','frozen_utc':data['frozen_utc'],'train_real_source_counts':dict(counts),'weights':data['source_weights']}),flush=True)


def verify(args):
    f=read(HERE/'FREEZE.json');assert f['helper_sha256']==codehashes()
    assert all(sha(p)==h for p,h in f['inputs_sha256'].items())
    source_root=args.source;assert git(source_root,'rev-parse','HEAD')==DOC
    assert sha(source_root/'CES_FL/src/ces_fl_head_ablation.py')==CORE
    expected=read(source_root/'CES_FL/paper/evidence/VALIDATION_SELECTED.json')
    expected_names={'S':'rf_s_s17','S_s7':'rf_s_s7','S_s27':'rf_s_s27','SJ_gentle':'rf_sjg_s17','SJ_gentle_s7':'rf_sjg_s7','SJ_gentle_s27':'rf_sjg_s27'}
    import torch  # CPU deserialization of trusted local checkpoint; no model construction/CUDA query.
    torch.set_num_threads(2)
    bindings=[];hashes=registry([TRAIN,SELECT,source_root/'CES_FL/src/ces_fl_head_ablation.py'])
    for name,arm in STUDIES:
        root=RUNS/arm;out=root/'out';c=read(out/'config.json');run=read(out/'RUN.json');r=read(out/'RUN_VERIFIED.json');pre=read(root/'preflight/PREFLIGHT.json')
        assert c==read(root/'config.json')
        assert digest(c)==run['config_digest']==r['config_digest']==pre['config_digest']
        assert c['source_sha256']==run['head_ablation_sha256']==pre['source_sha256']==CORE
        assert c['train']==str(TRAIN) and c['val']==str(SELECT)
        assert sha(TRAIN)==c['train_sha256'] and sha(SELECT)==c['val_sha256']
        src=r['rf_source_sha256'];assert src==run['rf_source_sha256'] and src in HISTORY
        raw=subprocess.check_output(['git','-C',str(source_root),'show',HISTORY[src]+':CES_FL/src/ces_fl_realfpr.py'])
        assert hashlib.sha256(raw).hexdigest()==src
        assert r['status']=='COMPLETED_VERIFIED' and r['epochs_completed']==10 and (out/'DONE').exists()
        assert r['selected_checkpoint_sha256']==expected[expected_names[arm]]['selected_checkpoint_sha256']
        cksha=sha(out/'best.pt');assert cksha==r['selected_checkpoint_sha256']
        ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
        assert ck['config']==c and ck['epoch']==r['selected_epoch'] and ck['validation']==r['validation']
        threshold=float(ck['calibration']['threshold_logit']);assert math.isfinite(threshold)
        assert pre['status']=='PASS' and not pre['test_read'] and pre['train_rows']==218266 and pre['val_rows']==31107
        inv=root/'preflight/inventory_PRIVATE.json';assert sha(inv)==pre['inventory_sha256']
        score=out/'validation_scores_PRIVATE.json';assert score.is_file()
        paths=[out/'best.pt',out/'RUN.json',out/'RUN_VERIFIED.json',out/'config.json',root/'config.json',root/'preflight/PREFLIGHT.json',inv,score]
        hashes.update(registry(paths))
        bindings.append({'name':name,'arm':arm,'run_directory':str(out),'config_digest':digest(c),
            'config_file_sha256':sha(out/'config.json'),'checkpoint_sha256':cksha,'selected_epoch':ck['epoch'],
            'threshold_logit':threshold,'rf_source_sha256':src,'historical_source_commit':HISTORY[src],
            'head_ablation_source_sha256':CORE,'inventory_sha256':pre['inventory_sha256'],'status':'PASS'})
        del ck
        print(json.dumps({'stage':'BINDINGS','name':name,'status':'PASS','selected_epoch':r['selected_epoch'],'historical_source':HISTORY[src]}),flush=True)
    write(HERE/'BINDINGS.json',{'status':'PASS','verified_utc':now(),'freeze_sha256':sha(HERE/'FREEZE.json'),
        'bindings':bindings,'input_sha256':hashes,'score_headers_and_alignment':'DEFERRED until analysis opens private outcomes',
        'checkpoint_deserialization':'CPU map_location; weights_only=True; no model execution','cuda_queried':False})


def analyze(args):
    f=read(HERE/'FREEZE.json');b=read(HERE/'BINDINGS.json')
    assert f['helper_sha256']==codehashes() and b['freeze_sha256']==sha(HERE/'FREEZE.json')
    assert all(sha(p)==h for p,h in b['input_sha256'].items())
    if (args.private/'SCORES_OPENED.json').exists():raise FileExistsError('Preserve diagnostic attempt; do not silently repeat')
    train=rows(TRAIN);val=rows(SELECT);ns=len(val);sources=[source(r) for r in val]
    if any(source(r) not in f['sources'] for r in val if r['label']==0):raise ValueError('SELECT real source provenance outside frozen mapping')
    # Graph is based solely on metadata, before private outcomes are opened.
    allcomp=connected_components(train+val);_,comp=np.unique(allcomp[len(train):],return_inverse=True)
    overlap={k:len({t for r in train for t in tokens(r[k])}&{t for r in val for t in tokens(r[k])}) for k in ('identity_ids','source_ids')}
    meta={'recorded_train_select_overlap_counts':overlap,'select_rows':ns,
          'real_source_counts':dict(collections.Counter(s for r,s in zip(val,sources) if r['label']==0)),
          'reference_source_counts_all_classes':dict(collections.Counter(sources)),
          'component_namespace':'All semicolon-separated recorded identities and masters, not first-identity convenience cluster.',
          'private_source_history_limit':'Only permitted TRAIN/SELECT manifests, cached inventories and builder code used. Mixed-split master manifests and raw TEST/CAL data were not opened.',
          'per_image_processing_history':'BLOCKED: acquisition labels are documented, but individual crop/config/master-byte histories are absent in the permitted split caches.',
          'biological_identity_separation_verified':False,'fake_source_label':'Recorded contributing-reference acquisition; not proof of fake pixel acquisition/evidence origin.'}
    write(args.private/'METADATA_COMPONENT_SUMMARY.json',meta)
    write(args.private/'SCORES_OPENED.json',{'utc':now(),'freeze_sha256':sha(HERE/'FREEZE.json'),'freeze_utc':f['frozen_utc'],
        'operation':'First private outcome parse in this diagnostic; metadata strata/weights already frozen.'})
    inventory_cache={};zs=[];thresholds=[];alignment=[];names=[]
    for item in b['bindings']:
        out=Path(item['run_directory']);d=read(out/'validation_scores_PRIVATE.json');cache=d['rows']
        assert d['split']=='validation' and d['manifest_sha256']==sha(SELECT) and len(cache)==ns
        assert d['checkpoint_sha256']==item['checkpoint_sha256']
        assert float(d['threshold'])==item['threshold_logit']
        assert d.get('test_read') is False and int(d['min_support'])==20
        invsha=item['inventory_sha256']
        if invsha not in inventory_cache:
            inv=read(out.parent/'preflight/inventory_PRIVATE.json')
            assert {x['split'] for x in inv}=={'train','val'}
            inventory_cache[invsha]={x['image']:x['sha256'] for x in inv if x['split']=='val'}
        inv=inventory_cache[invsha]
        fields=('gender','race','gender_race','age_bin','skin_bin','family','method','ids8')
        for r,v in zip(cache,val):
            assert posixpath.normpath(r['image'])==v['image'] and int(r['label'])==v['label']
            for k in fields:assert (str(r.get(k) or 'unknown').strip() or 'unknown')==v[k]
            for k in ('identity_ids','source_ids'):assert tokens(r[k])==tokens(v[k])
            assert r['image_sha256']==inv[v['image']]
        z=np.array(d['logits'],float);assert z.shape==(ns,) and np.isfinite(z).all()
        zs.append(z);thresholds.append(item['threshold_logit']);names.append(item['name'])
        alignment.append({'name':item['name'],'status':'PASS','rows':ns,'order_labels_demographics_contributors_and_inventory_hashes_match':True,
                          'frozen_threshold_exact_match':True,'cache_sha256':sha(out/'validation_scores_PRIVATE.json')})
        print(json.dumps({'stage':'ALIGNMENT','name':item['name'],'status':'PASS','rows':ns}),flush=True)
        del d,cache
    result=run_statistics(val,np.array(sources),comp,np.array(zs),np.array(thresholds),names,f,
                          progress=lambda n,total:print(json.dumps({'stage':'BOOTSTRAP','completed':n,'total':total}),flush=True))
    # Validate the unchanged native metrics against every selected run's stored aggregate report.
    for item in b['bindings']:
        report=read(Path(item['run_directory'])/'RUN_VERIFIED.json')['validation'];m=names.index(item['name'])
        for key,expected in [('AUC/overall',report['overall']['auc']),('FPR/overall',report['overall']['fpr']),
                             ('FNR/overall',report['overall']['fnr']),('D_native/gender_race',report['axes']['gender_race']['F_FPR'])]:
            assert math.isclose(result['point'][m,result['keys'].index(key)],expected,rel_tol=0,abs_tol=1e-12),(item['name'],key)
    np.savez_compressed(args.private/'BOOTSTRAP_AGGREGATE_DRAWS_PRIVATE.npz',draws=result['draws'])
    write(HERE/'ALIGNMENT.json',{'status':'PASS','private_outcomes_opened_after_freeze':True,'runs':alignment,
        'image_bytes_rehashed':False,'image_hash_scope':'Stored source-bound preflight inventory versus cached row hashes; no raw image reads/new predictions.'})
    write(HERE/'COVERAGE.json',dict(result['coverage'],metadata=meta,components=result['components']))
    write(HERE/'COHORT_SUPPORT.json',result['cohorts'])
    with (HERE/'METRICS_AND_INTERVALS.csv').open('w',newline='') as stream:
        cols=sorted({k for r in result['intervals'] for k in r});writer=csv.DictWriter(stream,fieldnames=cols,lineterminator='\n')
        writer.writeheader();writer.writerows(result['intervals'])
    write(HERE/'PRIMARY_RESULT.json',result['primary'])
    # Recheck file immutability after reading/analysis. No earlier artifacts are written.
    changed=[p for p,h in b['input_sha256'].items() if sha(p)!=h];assert not changed,'An input artifact changed during analysis'
    status='BLOCKED_PRIMARY_COMMON_SUPPORT' if result['primary']['estimate'] is None else result['primary']['status']
    write(HERE/'STOP_STATUS.json',{'status':status,'audit_execution':'COMPLETED_CPU_ONLY','stopped_utc':now(),
        'primary':result['primary'],'replicates':2000,'bootstrap_seed':17,'minimum_valid_replicates':1900,
        'all_matched_downstream_seeds':[17,7,27],'best_cell_selection':False,'new_predictions_generated':False,
        'scientific_source_configuration_or_guards_changed':False,'test_cal_accessed':False,'gpu_cuda_accessed':False,
        'input_artifacts_unchanged_after_analysis':True,'next_action':'STOP. No training or data revision authorized; report coverage/uncertainty to supervisor.'})
    print(json.dumps({'stage':'STOP','status':status,'primary':result['primary'],'components':result['components']['n']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','verify','analyze'])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--private',type=Path,required=True);a=p.parse_args()
    a.private.mkdir(parents=True,exist_ok=True)
    {'freeze':freeze,'verify':verify,'analyze':analyze}[a.stage](a)
