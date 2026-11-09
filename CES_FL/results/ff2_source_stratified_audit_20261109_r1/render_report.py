"""Aggregate-only presentation/checks; does not read private predictions or images."""
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text())
def csvfile(name,rows):
    fields=sorted({k for r in rows for k in r})
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def md(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def scaled(x):return f'{100*float(x):.4f}' if x not in ('',None) else 'BLOCKED'
def interval(r):
    if r['ci95_low'] not in ('',None):return '['+scaled(r['ci95_low'])+', '+scaled(r['ci95_high'])+']'
    return r['status']+' ('+str(r['valid_replicates'])+'/2000)'

freeze=read('FREEZE.json');bindings=read('BINDINGS.json');coverage=read('COVERAGE.json');stop=read('STOP_STATUS.json')
with (ROOT/'METRICS_AND_INTERVALS.csv').open() as f:metrics=list(csv.DictReader(f))
models=[r['name'] for r in bindings['bindings']];lookup={(r['model'],r['metric']):r for r in metrics}
coh={r['name']:r for r in read('COHORT_SUPPORT.json')};support=[];checks=0
for model in models:
    for name,c in coh.items():
        for metric in ['FPR','FNR','AUC']:
            r=lookup[model,metric+'/'+name]
            support.append(dict(r,cohort_kind=c['kind'],axis=c['axis'],group=c['group'],source=c['source'],
                n_real=c['n_real'],n_fake=c['n_fake'],n_real_components=c['n_real_components'],n_fake_components=c['n_fake_components']))
csvfile('GROUP_FAMILY_SOURCE_UTILITY.csv',support)
native=[];disparities=[];source_rows=[]
for model in models:
    b=next(x for x in bindings['bindings'] if x['name']==model)
    row={'model':model,'selected_epoch':b['selected_epoch'],'threshold_logit':b['threshold_logit'],
         'n_real':2438,'n_fake':28669,'n_components':coverage['components']['n']}
    for metric in ['AUC/overall','FPR/overall','FNR/overall','D_native/gender_race']:
        r=lookup[model,metric];row[metric]=r['estimate'];row[metric+'_ci95_low']=r['ci95_low'];row[metric+'_ci95_high']=r['ci95_high'];row[metric+'_interval_status']=r['status']
    native.append(row)
    for r in metrics:
        if r['model']==model and r['metric'].startswith(('D_native/','D_standardized/','native_minus_standardized/','within_source_D/','within_source_FPR_gap/','native_FPR_gap/','standardized_FPR/')):disparities.append(r)
    for c in coh.values():
        if c['kind']!='source_group':continue
        r=lookup[model,'FPR/'+c['name']]
        fp=round(float(r['estimate'])*c['n_real']) if c['n_real'] else 0
        source_rows.append(dict(r,axis=c['axis'],group=c['group'],source=c['source'],n_real=c['n_real'],false_positives=fp,
            n_real_components=c['n_real_components'],n_recorded_real_masters=c['n_real']))
    # Independently reconstruct native rates from source-count cells, with empty rates left undefined.
    for ax,groups in freeze['groups'].items():
        qs=[];fp_all=0;n_all=0
        for g in groups:
            rr=[r for r in source_rows if r['model']==model and r['axis']==ax and r['group']==g]
            n=sum(r['n_real'] for r in rr);fp=sum(r['false_positives'] for r in rr)
            assert n>0;qs.append(fp/n);fp_all+=fp;n_all+=n
        native_d=sum(abs(q-fp_all/n_all) for q in qs)
        assert math.isclose(native_d,float(lookup[model,'D_native/'+ax]['estimate']),abs_tol=1e-12)
        checks+=1
    # Gender is the one fully supported source-standardized axis here.
    gs=freeze['groups']['gender'];qstd=[];pi=[]
    for g in gs:
        rr=[r for r in source_rows if r['model']==model and r['axis']=='gender' and r['group']==g]
        assert all(r['n_real']>0 for r in rr)
        qstd.append(sum(freeze['source_weights'][r['source']]*r['false_positives']/r['n_real'] for r in rr))
        pi.append(sum(r['n_real'] for r in rr)/2438)
    pooled=sum(p*q for p,q in zip(pi,qstd));std=sum(abs(q-pooled) for q in qstd)
    assert math.isclose(std,float(lookup[model,'D_standardized/gender']['estimate']),abs_tol=1e-12);checks+=1
csvfile('NATIVE_UTILITY.csv',native);csvfile('DISPARITIES.csv',disparities);csvfile('SOURCE_BY_GROUP_REAL_FPR.csv',source_rows)
csvfile('PAIRED_CONTRASTS.csv',[r for r in metrics if r['kind']=='paired'])

for axis in freeze['groups']:
    rows=[]
    for group in freeze['groups'][axis]:
        for s in freeze['sources']:
            c=coh['source='+s+'/'+axis+'='+group]
            tr=next(x['n_real'] for x in freeze['train_source_group_support'] if x['axis']==axis and x['group']==group and x['source']==s)
            rows.append([group,s,tr,c['n_real'],c['n_real_components']]+[scaled(lookup[m,'FPR/'+c['name']]['estimate']) for m in models])
    (ROOT/('SOURCE_'+axis.upper()+'_FPR.md')).write_text('# SELECT real support and FPR by source × '+axis+'\n\nAll rates below are percentages. Empty cells are BLOCKED, not zero FPR. Components use all recorded donor/master links, not biological person counts. Intervals/status/denominators are in SOURCE_BY_GROUP_REAL_FPR.csv.\n\n'+md(['Group','Source','TRAIN real','SELECT real','SELECT real components']+models,rows)+'\n')

rows_native=[]
for m in models:
    rows_native.append([m,scaled(lookup[m,'AUC/overall']['estimate']),scaled(lookup[m,'FPR/overall']['estimate']),
        scaled(lookup[m,'FNR/overall']['estimate']),scaled(lookup[m,'D_native/gender_race']['estimate']),
        interval(lookup[m,'AUC/overall']),interval(lookup[m,'FNR/overall'])])
rows_gender=[[m,scaled(lookup[m,'D_native/gender']['estimate']),scaled(lookup[m,'D_standardized/gender']['estimate']),
    scaled(lookup[m,'native_minus_standardized/gender']['estimate']),interval(lookup[m,'native_minus_standardized/gender'])] for m in models]
rows_source=[[m,scaled(lookup[m,'FPR/source=FFHQ']['estimate']),scaled(lookup[m,'FPR/source=FairFace']['estimate']),
    scaled(lookup[m,'within_source_D/gender_race/FairFace']['estimate']),interval(lookup[m,'within_source_D/gender_race/FairFace'])] for m in models]
pairs=[]
for seed in [17,7,27]:
    name='SJg-minus-S_s'+str(seed)
    pairs.append([seed,scaled(lookup[name,'D_native/gender_race']['estimate']),interval(lookup[name,'D_native/gender_race']),
        scaled(lookup[name,'AUC/overall']['estimate']),interval(lookup[name,'AUC/overall']),
        scaled(lookup[name,'FNR/overall']['estimate']),interval(lookup[name,'FNR/overall'])])
text='''# FairForge-2 source-stratified frozen-score audit

**Final stop status: BLOCKED for the primary gender×race source-standardization estimand; INCONCLUSIVE for the relevant fairness intervals. CPU execution completed. STOP—no training or data revision follows.**

This is exploratory SELECT analysis of already selected, unchanged-image RF-S/RF-SJg exports. It is not a new model, an improved decision rule, or independent confirmation. All matched seeds17/7/27 are reported; no best-cell selection.

## Primary result and support

The fixed TRAIN source weights are FFHQ=0.18930678905644632 and FairFace=0.8106932109435536, from3176/13601 real TRAIN images. SELECT has242/2196 reals, respectively. They were frozen before private prediction outcomes were parsed. Full race and gender×race standardization is not identified: SELECT has zero FFHQ-real examples labelled Others, including both Female-Others and Male-Others. The same cells are absent in TRAIN. No source/group was discarded, imputed or renormalized away. The missing counterfactual source cells cannot be learned from existing scores.

Therefore the primary RF-S seed17 contrast D_native−D_standardized has **no point estimate or confidence interval**. This remains blocked for both methods at all three seeds. The full-group within-FFHQ race/joint estimands are likewise unavailable; observed cell rates remain visible.

## Source and input verification

Documentation handoff: `76dcb54e240dcf56475dbf587836cdadcdad5aab`, documentation-only over reviewed scientific commit `e7fe0ee03e6cfacca81c558bbf632d2f72e80bc0`. A separate clean detached source checkout was used. Scientific source/configuration/guard files were unchanged.

All six actual checkpoint byte hashes, actual config dictionaries/digests, source/core hashes, preflight inventory bindings, selected epochs and CPU-loaded checkpoint thresholds passed. RF-S seed17 was verified against its historical source0a135a0/SHA5ee7c8b, rather than forced through the later schema; the other runs match ac6cba4/SHA2d99295. Cache rows match the full31107-row SELECT manifest in order, labels, demographic fields, method/family, complete recorded identity/source-ID sets and source-bound cached image hashes. Every stored global threshold matched the selected checkpoint exactly. Native metrics replay from cached scores matched the saved reports to1e-12.

Acquisition labels come from documented canonical real-source keys and exact source-key/own-image matches, not appearance. Master/person/donor identifiers are recorded metadata, not independently verified biological identities. Only TRAIN/SELECT manifests, their cached inventories, SELECT exports and relevant builder source code were used. Mixed-split source-master manifests, raw image pixels, TEST/CAL data and Ding models were not accessed. Individual crop/config/master-byte histories are not present in these permitted split caches: a finer processing-stratified or causal history decomposition is **BLOCKED**, not silently inferred from the dataset name. Fake source breakdowns below refer to recorded contributing-reference acquisition, not proof of pixel/evidence origin.

## Native unchanged-image utility

Supplementary age/skin utility rows use inherited automatic estimates. Unknown skin values remain explicit; no demographic labels were generated or inferred from appearance. Intervals condition on recorded labels and do not account for annotation error. The full metadata check also matched every one of the16 CSV fields against all six exports, including label provenance and proxy values.

All rates/AUC/disparities are multiplied by100. `100D` is the sum of absolute group FPR deviations, not an overall FPR or a percentage explained. AUC/FNR intervals below are recorded-component, conditional on the already selected checkpoints; they do not include selection or encoder-pretraining uncertainty.

'''+md(['Model','AUC','FPR','FNR','Gender×race 100D','AUC95% CI','FNR95% CI'],rows_native)+'''

## Identified gender-only standardization, with inconclusive intervals

Gender has both acquisition sources in both groups, so its point estimand is identified. Source weights remain fixed from TRAIN; the same SELECT real-group proportions pool native/standardized rates. These are additional exploratory estimates, not replacements for native model rows.

'''+md(['Model','Native gender100D','Standardized gender100D','Native−standardized','95% CI/status'],rows_gender)+'''

Only1871/2000 component replicates defined all required gender/source cells, below the fixed1900 requirement. The apparent point reductions therefore do not support a statistically resolved source-composition association. The threshold was not retuned.

## Source rates and residual within-source disparity

'''+md(['Model','FFHQ real FPR%','FairFace real FPR%','Within-FairFace joint100D','Joint residual CI/status'],rows_source)+'''

FairFace has all eight joint groups and a computable residual disparity. Its full joint interval only has202/2000 valid draws, so residual differences are inconclusive. FFHQ lacks the two Others cells: its full eight-group residual is blocked rather than recomputed on a favorable subset. The augmentation lowers FFHQ false alarms but raises FairFace false alarms in these native point estimates. Source sensitivity and this observational association do not establish that compression alone causes demographic disparity or quantify a causal fraction explained.

[Gender/source support and rates](SOURCE_GENDER_FPR.md) · [Race/source](SOURCE_RACE_FPR.md) · [Gender×race/source](SOURCE_GENDER_RACE_FPR.md). These preserve all empty cells. Native per-group/family/source AUC/FNR, fake support, component counts and interval status are in GROUP_FAMILY_SOURCE_UTILITY.csv. Source/family cells without both classes remain undefined; no fake-source standardized aggregate is manufactured.

## Exact paired SJg-minus-S contrasts, every seed

'''+md(['Seed','Native joint Δ100D','Joint Δ CI/status','ΔAUC pp','AUC Δ95% CI','ΔFNR pp','FNR Δ95% CI'],pairs)+'''

Native joint point differences favor SJg at all three seeds, but their intervals only have999/2000 valid draws. Utility AUC/FNR differences have2000 valid draws and indicate a detection cost descriptively. All additional intervals are unadjusted/exploratory; no significance or method selection is inferred from them. Standardized race/joint paired contrasts are blocked by the same missing cells. Standardized gender paired contrasts are reported with their inconclusive interval status in PAIRED_CONTRASTS.csv.

## Why image counts overstate uncertainty support

The all-contributor/master dependency graph projects to45 SELECT-bearing components, largest2787 rows. These are **dependency components, not45 biological people**. Many real source×joint cells have only one or two recorded components because fakes join multiple contributors. Resampling every frame independently would discard that dependence. Recorded TRAIN/SELECT identity/source-token overlap counts were both zero; this does not prove biological-person separation.

The prespecified2,000-draw bootstrap uses analysis seed17, whole recorded connected components, the same draws for all six models and both estimands, fixed decisions and fixed eligible groups. Acquisition remains an analytic stratum rather than two bootstrap units. Whole components are resampled globally, preserving any cross-stratum links; this choice was explicitly frozen before private outcomes. TRAIN source weights remain fixed; SELECT group proportions are re-estimated together in both views within each replicate. No component, group or difficult draw was removed to meet the95% rule.

Primary race/joint standardization is structurally unidentified even before bootstrapping. Other intervals are withheld when fewer than1900 draws retain the necessary denominators. Missing support and interval failures must not be converted into a declaration that source composition dominates, that within-source bias is absent, or that we improved fairness.

## Explicit stop and files

**STOP at this report.** No new predictions, labels, images, GPU/CUDA operation, threshold/group calibration, scientific code/configuration/approval/guard change, new Ding computation, or TEST/CAL access. No trained arm was added. A training or data-revision proposal requires separate review.

- FREEZE.json records source definitions, TRAIN weights/support, contrast/bootstrap/multiplicity rules and the pre-score timestamp.
- BINDINGS.json and ALIGNMENT.json record the six actual artifact checks and input SHA256s.
- METRICS_AND_INTERVALS.csv contains all native/standardized/residual/group/family metrics and validity denominators; derivatives are provided as smaller CSVs.
- COVERAGE.json and COHORT_SUPPORT.json document missing cells, source provenance and recorded-component support.
- PRIMARY_RESULT.json and STOP_STATUS.json are the machine-readable blocked/inconclusive conclusion.
- audit.py, audit_stats.py, test_audit.py and CPU_CHECKS.log preserve the scoped helper and meaningful checks. No trainer is modified.
- Private bootstrap arrays and opening-event metadata remain under `/home/sachin.chaudhary/ces_fl_runs/ff2_source_stratified_audit_20261109_r1/`; no private rows, faces, prediction vectors or checkpoints are published.

This analysis cannot certify independent person-level links or processing histories absent from the permitted metadata. It conditions on previously selected checkpoints and shared pretraining, uses exposed SELECT for exploration, and supplies no independent confirmation or overall superiority claim.
'''
text=text.replace('All rates/AUC/disparities are multiplied by100.',
    'Markdown tables multiply rates/AUC/disparities by 100; raw machine-readable CSV estimates and intervals remain fractions.')
for a,b in {'seeds17/7/27':'seeds 17/7/27','seed17':'seed 17','from3176/13601':'from 3,176/13,601',
            'has242/2196':'has 242/2,196','source0a135a0':'source 0a135a0','full31107-row':'full 31,107-row',
            'to1e-12':'to 1e-12','Only1871/2000':'Only 1,871/2,000','fixed1900':'fixed 1,900',
            'has202/2000':'has 202/2,000','have999/2000':'have 999/2,000','have2000':'have 2,000',
            'to45 SELECT':'to 45 SELECT','largest2787':'largest 2,787','not45':'not 45',
            'prespecified2,000':'prespecified 2,000','the95%':'the 95%','than1900':'than 1,900'}.items():text=text.replace(a,b)
(ROOT/'REPORT.md').write_text(text)
(ROOT/'UNITS.json').write_text(json.dumps({'raw_csv_estimate_and_interval':'fractions; summed disparity can exceed1',
    'markdown_rates_and_auc':'raw fractions multiplied by100','counts':'integers','threshold_logit':'unchanged raw logit',
    'unidentified_rates':'empty/null/BLOCKED, never imputed zero','percentage_explained':'not computed'},indent=2)+'\n')
(ROOT/'PRESENTATION_CHECKS.json').write_text(json.dumps({'status':'PASS','independent_native_and_gender_standardization_reconstructions':checks,
    'all_models':models,'full_joint_standardization_still_blocked':all(x['estimate']=='' for x in metrics if x['metric']=='D_standardized/gender_race'),
    'bootstrap_valid_threshold_retained':1900,'no_new_model_or_prediction_computation':True},indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':checks,'source_real_rows':len(source_rows),'utility_rows':len(support),'primary_stop':stop['status']},indent=2))
