"""Render the final cohort report from aggregate-only public artifacts."""
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text())
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
with (ROOT/'EXTENDED_VALIDATION_METRICS.csv').open() as f:summary=list(csv.DictReader(f))
with (ROOT/'VALIDATION_SOURCE_REAL_FPR.csv').open() as f:sources=list(csv.DictReader(f))
with (ROOT/'EPOCH_SUMMARY.csv').open() as f:epochs=list(csv.DictReader(f))
with (ROOT/'PAIRED_CONTRIBUTOR_INTERVALS.csv').open() as f:intervals=list(csv.DictReader(f))
obj=read('FULL_TRAIN_OBJECTIVE_DIAGNOSTIC.json');verify=read('FINAL_VERIFICATION.json');gpu=read('COHORT_ALLOCATION_END.json')
def row(model,axis='gender_race',view='VAL10'):return next(x for x in summary if x['model']==model and x['axis']==axis and x['view']==view)
def n(x):return f'{100*float(x):.4f}'
models=['teacher','control','treatment'];native=[];fixed=[]
for name in models:
    for dest,view in [(native,'VAL10'),(fixed,'fixed0.5')]:
        x=row(name,view=view);dest.append([name,'existing' if name=='teacher' else 1,n(x['overall_auc']),n(x['overall_fpr']),n(x['overall_fnr']),n(x['F_FPR']),n(x['worst_group_auc'])])
axes=[]
for axis in ['gender','race','gender_race','age_bin','skin_bin','ids8','age_x_skin16','coarse_gender_age_race8']:
    x,y=row('control',axis),row('treatment',axis)
    delta=100*(float(y['F_FPR'])-float(x['F_FPR']))
    axes.append([axis,n(row('teacher',axis)['F_FPR']),n(x['F_FPR']),n(y['F_FPR']),f'{delta:+.4f}',
        'worse' if delta>1e-12 else 'lower point estimate' if delta< -1e-12 else 'unchanged'])
sr=[]
for name in models:
    x={r['source']:r for r in sources if r['model']==name and r['group']=='ALL' and r['view']=='VAL10'}
    sr.append([name,x['FFHQ']['false_positives']+'/242',n(x['FFHQ']['fpr']),x['FairFace']['false_positives']+'/2196',n(x['FairFace']['fpr'])])
epochrows=[[r['arm'],r['epoch'],r['feasible'],n(r['auc']),n(r['fnr']),n(r['joint_F_FPR']),f"{float(r['classification']):.6f}",f"{float(r['source_tail']):.6f}",f"{float(r['fake_guard']):.6f}"] for r in epochs]
ci=[]
for r in intervals:
    ci.append([r['metric'],f"{100*float(r['estimate']):+.6f}",
        '['+n(r['ci95_low'])+', '+n(r['ci95_high'])+']' if r['ci95_low'] else 'withheld',r['valid_replicates'],r['status']])
red=100*(obj['control']['source_tail']-obj['treatment']['source_tail'])/obj['control']['source_tail']
stage_names=['qualify_profile','prepare','train_control','train_treatment','export_control','export_treatment']
timing=[[s,f"{(read(s+'.RECEIPT.json')['end_node_unix']-read(s+'.RECEIPT.json')['start_node_unix'])/60:.3f}",read(s+'.RECEIPT.json')['status']] for s in stage_names]
text='''# Source-tail head-training cohort: completed results

**Execution/export verdict: PASS. Scientific conclusion: no meaningful fairness benefit established; primary fairness uncertainty INCONCLUSIVE. Stop after this cohort—no automatic extension or tuning.**

Both exact three-epoch seed-17 arms completed, both selected epoch 1 under the unchanged rule, and both reconstructed full students passed exact native SELECT replay. All 274 encoder parameter/BatchNorm tensors match the teacher bit for bit. The source-tail coefficient changed learned head parameters and reduced its TRAIN surrogate relative to matched control, but its hard validation fairness effect is tiny and not statistically resolved. This does not beat Ding or validate the full L_H/counterfactual method.

## Native SELECT comparison

All rates, AUC and summed disparity below are multiplied by 100. `100F_FPR` is a sum, not overall FPR. Rows are coherent selected checkpoints, not best cells across epochs/seeds. There are 31,107 SELECT images (2,438 real / 28,669 fake). Teacher and students use their own already declared global validation-real FPR10 calibration; no group-specific threshold is introduced.

'''+table(['Model','Selected epoch','AUC%','FPR%','FNR%','Gender×race100F_FPR','Worst-group AUC%'],native)+'''

Treatment minus control: AUC −0.000449 percentage points; FNR unchanged at0.289511%; primary100F_FPR −0.029543 points (about0.085% relative). The teacher-to-control change must not be credited to source-tail training: both downstream arms share further BCE+fake-score-guard adaptation, and only treatment adds the source term.

**Fixed probability0.5 view, kept separate:**

'''+table(['Model','Selected epoch','AUC%','FPR%','FNR%','Gender×race100F_FPR','Worst-group AUC%'],fixed)+'''

At fixed0.5 the treatment has lower FPR but higher FNR than control. These decisions differ from the primary validation-calibrated operating point; the better-looking view was not used to select a checkpoint or substitute a result.

## All reported demographic axes and regressions

'''+table(['Axis','Teacher100F_FPR','Control100F_FPR','Treatment100F_FPR','Treatment−control','Point reading'],axes)+'''

Gender/age disparity does not change; race/joint point reductions are very small. Skin, inherited `ids8`, and the explicitly defined coarse gender×age×race eight-group view worsen. Age×skin point disparity decreases, while its worst-group AUC declines. `ids8` is the inherited model-estimated label, not the coarse eight-group axis. Age/skin labels are estimates and unknowns remain in native totals. Extra-intersection estimates are descriptive point-only; no newly selected objective/threshold is justified by a favorable cell. Full per-group FPR/FNR/AUC/support and both fixed0.5/VAL10 views are in the published CSVs.

## Acquisition-source false alarms and TRAIN surrogate

'''+table(['Model','FFHQ false positives','FFHQ FPR%','FairFace false positives','FairFace FPR%'],sr)+'''

The treatment has one more FFHQ false positive and one fewer FairFace false positive than the control at their stored validation-calibrated thresholds. The hard source gap is not improved despite the smoothed TRAIN penalty change. This is consistent with the limitations of a fixed teacher-tail surrogate and changed score calibration; it does not establish a unique failure mechanism.

Full-TRAIN descriptive source-tail penalties at the selected heads:

'''+table(['Model','Full-TRAIN source-tail penalty','Full-TRAIN fake-score guard'],[[m,f"{obj[m]['source_tail']:.9f}",f"{obj[m]['fake_guard']:.9f}"] for m in models])+f'''

Treatment penalty is {red:.2f}% below the control, but remains above the original teacher. These are in-sample TRAIN quantities computed by CPU head arithmetic on existing cached features, not new encoder inference or independent evidence. The fake-score guard discourages fake-score decreases; it does not guarantee zero fake errors. References stay fixed at TRAIN-real85/90/95 score quantiles, teacher-real IQR scale and pooled soft risks. All family/source FNR and utility rows are included rather than hiding regressions.

No source-standardized race/joint metric is manufactured: FFHQ-real Female-Others and Male-Others remain absent. Source breakdowns of fakes refer to recorded contributing-reference acquisition, not proven pixel origin.

## Epoch histories and actual exposure

'''+table(['Arm','Epoch','Feasible','AUC%','FNR%','Joint100F_FPR','BCE step mean','Source-tail step mean','Fake guard step mean'],epochrows)+'''

Every arm completed20,463 optimizer updates:6,821 per epoch, all218,266 original TRAIN rows once per epoch (654,798 original-image exposures per arm). Both used the same deterministic auxiliary draws:16 reals from each source per update. That is109,136 auxiliary exposures/source/epoch, or654,816 auxiliary exposures across both sources over three epochs per arm. Sampling reuse is not unique-source coverage or more independent identities.

Reconstructing the exact auxiliary schedule from the fixed seed confirms every FFHQ real appears in each epoch; FairFace unique rows seen were13,595 /13,599 /13,596 of13,601. Average source reuse is34.36 exposures per FFHQ real per epoch versus8.02 per FairFace real. Per-source×joint-group exposure, including structural zero cells, and actual unique coverage are in AUXILIARY_SOURCE_JOINT_EXPOSURE.csv and ACTUAL_AUXILIARY_COVERAGE.csv. Main BCE coverage was identical across arms; no difficult original was excluded.

## Paired recorded-contributor uncertainty

The 2,000-draw analysis uses seed17 and the audited45 SELECT-bearing recorded identity/master/donor components, same draws for both selected heads, fixed thresholds and fixed group eligibility. It requires at least1,900 valid draws. These are dependency components, not45 biological people; derived frames are not treated as independent subjects. Intervals condition on the selected checkpoints and shared pretrained encoder and do not account for repeated validation selection or label error. All intervals are exploratory and unadjusted across axes.

'''+table(['Treatment−control metric','Difference×100','95% interval×100','Valid /2000','Status'],ci)+'''

The primary joint-disparity CI is withheld because only999/2000 draws retain all joint-group denominators. It is INCONCLUSIVE; neither significance nor zero effect is established. The other reported intervals span zero. Sparse support is not repaired by dropping cells, splitting components or changing the95% rule.

## Exact implementation and verification

Scientific commit: **9bb8e8332b8a3aa78d83f74ef21bca0b7dcd6754**, clean detached source at `/home/sachin.chaudhary/ces_fl_sources/cesfl-source-tail-shared-9bb8e83-20261109`. Complete trained L_A head is retained at initialization; only its2,049 parameters are optimized. One Xception encoder is frozen and shared through a float32 TRAIN/SELECT cache. No original L_H probe/alignment terms, JPEG or synthetic demographic edits are included. Batch32, seed17, three epochs, AdamW headLR1e-4, weight_decay1e-4, clip5, lambda_fake1; source coefficient0 versus0.1 is the only arm change.

- Supplied CPU synthetic pipeline checks PASS, including resume identity, objective gradients, cache tampering and feature/native-forward equivalence. Actual CUDA fixture lifecycle and full224 feature/native-forward checks PASS; neither is a benchmark result.
- Actual teacher hash, source sidecar/provenance and manifest bindings PASS; TRAIN-real source labels exactly3176FFHQ/13601FairFace. Source labels derive from canonical recorded keys, not appearance. No mixed TEST/CAL master bank was used.
- Shared production cache hashes PASS; teacher feature/head/native forward max difference0 and exact stored native validation replay PASS. The cache contains218266×2048 TRAIN and31107×2048 SELECT float32 pooled features, with checkpoint-bound image SHA checking during every actual image read.
- Independent outcome verification matches all nine cache blob hashes, exact TRAIN reference recomputation, complete teacher head, fixed epoch-selection logic, exact actual steps/auxiliary schedule, selected checkpoint bytes, full-export head and274 unchanged encoder/BN tensors. Both exports record source labels unnecessary at inference.
- A presentation-only float32/float64 comparison issue was caught before publication: comparing cached float32 teacher logits directly against a nextafter float64 threshold rounded the boundary. Reporting now explicitly uses float64 decisions as the native evaluator does. The final teacher and both student metrics independently match native reports. The initial presentation attempt is preserved privately; no scientific cache/config/result or threshold was altered.

All exact source/config/cache/head/full-student hashes are in FINAL_VERIFICATION.json, CACHE_VERIFIED.json, *_RUN_VERIFIED.json and *_EXPORT_VERIFIED.json. Full-weight/feature/row-score files remain private.

## Compute and operational accounting

'''+table(['Stage','Wall minutes while one GPU lease held','Outcome'],timing)+f'''

The actual complete lease was **{gpu['held_gpu_hours']:.6f} GPU-hours ({gpu['held_gpu_hours']*60:.3f} minutes)**, including qualification, loading, I/O, cache, both head runs, exports, held idle and bookends; the cap was2GPU-hours and was not exceeded. Device: gpu01 physical GPU6, H10080GB, UUIDGPU-61f54ca7-3cee-1c93-def3-118523e95241. Existing user-authorized shared-pool-v2 capacity lease, not an exclusive/PBS allocation claim; all CUDA commands were bound to that one UUID and inside the existing ledger/deadline wrapper. No other job was displaced. No source-tail process or ledger reservation is active now.

Historical missing inference was conservatively charged3.227956GPU-hours over the full recorded first-to-last three-card window. This is an upper-bound accounting charge, not fabricated measured actual GPU time. Existing older manual measured charges were preserved and their missing state fields normalized for the strict ledger wrapper. Total shared charge now **{verify['shared_charged_gpu_hours']:.6f}/72**, leaving **{verify['remaining_shared_gpu_hours']:.6f} recorded GPU-hours**; none is automatically granted to another cohort. No original L_H budget was transferred.

Initial launch failed before leasing because a linked-worktree .git path lived in login-node-local /tmp. It was preserved and replaced by a self-contained shared detached checkout of the identical commit; source checks were not disabled. An idle-only capacity attempt also stopped before acquisition. Both consumed zeroGPU lease time. The successful shared-pool route is documented separately, with actual binding/lease evidence.

## Final decision and stop

**Software/execution/export PASS; source-tail fairness contribution remains unproven and the primary interval INCONCLUSIVE. Do not scale or retune automatically.** The treatment meets the fixed utility guards but has virtually the same selected primary disparity as its matched control, some regressions, and no identified improvement in hard source FPR. There is no evidence from this cohort that we beat Ding. No new Ding computation or TEST/CAL images, predictions, tuning or evaluation occurred; only TRAIN/SELECT were scored in the authorized stages. Historical execution metadata was inspected solely to reconcile accounting.

The cohort is finished. Existing logs/results, frozen scientific source and all private artifacts are preserved. Any new study requires a separate reviewed proposal; remaining hours are not permission for another candidate.
'''
for a,b in {'at0.289511%':'at 0.289511%','about0.085%':'about 0.085%','TRAIN-real85/90/95':'TRAIN-real 85/90/95',
    'completed20,463':'completed 20,463',':6,821':': 6,821','all218,266':'all 218,266','is109,136':'is 109,136',
    'or654,816':'or 654,816','were13,595':'were 13,595','of13,601':'of 13,601','is34.36':'is 34.36',
    'versus8.02':'versus 8.02','seed17':'seed 17','audited45':'audited 45','least1,900':'least 1,900',
    'not45':'not 45','only999/2000':'only 999/2000','the95%':'the 95%','its2,049':'its 2,049',
    'Batch32':'Batch 32','headLR1e-4':'head LR 1e-4','weight_decay1e-4':'weight_decay 1e-4','clip5':'clip 5',
    'lambda_fake1':'lambda_fake 1','coefficient0 versus0.1':'coefficient 0 versus 0.1','full224':'full 224',
    'exactly3176FFHQ/13601FairFace':'exactly 3,176 FFHQ /13,601 FairFace','difference0':'difference 0',
    'contains218266×2048':'contains 218266×2048','and31107×2048':'and 31107×2048','and274':'and 274',
    'was2GPU-hours':'was 2 GPU-hours','GPU6':'GPU 6','H10080GB':'H100 80GB','UUIDGPU-':'UUID GPU-',
    'charged3.227956GPU-hours':'charged 3.227956 GPU-hours','zeroGPU':'zero GPU'}.items():text=text.replace(a,b)
(ROOT/'REPORT.md').write_text(text)
status='''# Source-tail cohort: COMPLETED, STOPPED

Both exact three-epoch arms completed and both full students passed exact native SELECT export replay. Each selected epoch1. No source-tail process or budget reservation remains active. The earlier RUNNING entries below are historical snapshots, not live state.

| Arm | AUC% | FNR% | Gender×race100F_FPR | Execution/export |
|---|---:|---:|---:|---|
| Control |99.8073|0.2895|34.8501|COMPLETED_VERIFIED /PASS|
| Treatment |99.8069|0.2895|34.8206|COMPLETED_VERIFIED /PASS|

Treatment's primary point gain is only0.02954points; joint interval INCONCLUSIVE (999/2000 valid,1900 required). The TRAIN source-tail surrogate decreases7.87% versus control but remains above teacher. Skin and coarse-intersection metrics regress; gender/age unchanged. No meaningful fairness improvement or win over Ding established.

Actual one-GPU lease charge0.401480GPU-h includes all checks/cache/arms/export/idle/bookends, within2h. Reconciled shared ledger68.230538/72 charged,3.769462 remains; no extension authorized. [Full report](REPORT.md), [final verification](FINAL_VERIFICATION.json), [all metrics](EXTENDED_VALIDATION_METRICS.csv), [paired uncertainty](PAIRED_CONTRIBUTOR_INTERVALS.csv).

**STOP.** No training retry, parameter sweep, new dataset, image generation, TEST/CAL access or Ding computation is attached. Scientific commit/config unchanged. Images, features, row scores and model binaries remain private.

---

'''
p=ROOT/'STATUS.md';old=p.read_text();p.write_text(status+old)
(ROOT/'DECISION.json').write_text(json.dumps({'cohort':'COMPLETED','execution_export':'PASS','scientific_conclusion':'NO_MEANINGFUL_FAIRNESS_BENEFIT_ESTABLISHED',
    'primary_uncertainty':'INCONCLUSIVE_COMPONENT_SUPPORT','selected_epoch_control':1,'selected_epoch_treatment':1,
    'primary_difference_fraction':float(row('treatment')['F_FPR'])-float(row('control')['F_FPR']),
    'cohort_gpu_hours':gpu['held_gpu_hours'],'automatic_extension':False,'next_action':'STOP; separate reviewed proposal required.'},indent=2)+'\n')
print(json.dumps({'status':'REPORT_COMPLETE','native_models':3,'demographic_axes':8,'cohort_gpu_hours':gpu['held_gpu_hours'],'decision':'STOP_NO_FAIRNESS_WIN_CLAIM'},indent=2))
