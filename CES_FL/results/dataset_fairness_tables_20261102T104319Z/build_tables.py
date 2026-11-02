"""Reporting only: count receipts and previously calculated aggregate metrics.
No image decode, score-array loading, calibration, model inference or training.
"""
import csv,datetime,hashlib,json,math,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-tail-response-20261102-r1')
ARCHIVE=Path('/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents')
SOURCES={}
def load(path):
 p=Path(path).resolve();data=p.read_bytes();SOURCES[str(p)]=hashlib.sha256(data).hexdigest();return json.loads(data)
splits=load(ROOT/'SPLITS.json')
p=load('/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1/training/RUN_VERIFIED.json')
assert p['status']=='PASS' and p['native_independent_validation_replay'] and not p['test_evaluated']
assert p['validation']['units']=='fractions' and p['validation']['overall']['n']==31107
aggregate=load(REPO/'CES_FL/results/upgrade_tests_s17_20261101T040407Z/aggregate.json')
legacy=load(ARCHIVE/'CESEL_MA/runs_ff2public/cfesd_s7/eval/ff2_test.json')
legacycal=load(ARCHIVE/'CESEL_MA/runs_ff2public/cfesd_s7/eval/calibration.json')
leakfree=load(ARCHIVE/'CESEL_MA/runs_ff2public/cfesd_s7/eval/ff2_test_leakfree_video.json')
ding_evidence=load(REPO/'CES_FL/proposals/ff2_indomain_s17_20261101T042828Z/ding_baseline_evidence_20261101.json')
ding_exports=[{'file':f.name,'data':load(f)} for f in sorted((REPO/'DINGREPRO_LIN/results/eval_ding_protocol').glob('*ff2*.json'))]
TRANSFERS={}
public_test_sha=next(r['sha256'] for r in splits['records'] if r['dataset']=='FairForge-2 public' and r['split']=='test')
for arm,a in aggregate['arms'].items():
 x=a['datasets']['fairforge2'];assert x['state']=='DONE' and x['verification']['status']=='PASS'
 assert x['metrics']['manifest_sha256']==public_test_sha and x['verification']['rows']==87564
 assert not x['verification']['test_calibration_or_selection']
 for mode in ['calibrated','fixed_probability_0_5']:
  assert x['metrics'][mode]['units']=='fractions' and x['metrics'][mode]['overall']['n']==87564
 TRANSFERS[arm]=x
assert legacy['n']==87564 and abs(legacy['auc']-98.25976224316973)<1e-9
# Verify historical percent-vs-fraction conventions using stored group aggregates.
for mode in ['ding@0.5','fair@val']:
 for axis in ['gender','race','intersection']:
  report=legacy[mode];g=report[axis]
  ffpr=100*sum(abs(v['FPR']-report['overall']['FPR']) for v in g['groups'].values())
  assert abs(ffpr-g['F_FPR'])<1e-8,(mode,axis,'native unit mismatch')
AXES=['gender','race','gender_race','age_bin','skin_bin','ids8']
NAMES={'gender':'Gender','race':'Race','gender_race':'Gender x race','age_bin':'Age bins','skin_bin':'Skin bins (ITA)','ids8':'Native ids8'}
SHORT={'R0_frozen_bce':'R0','R1_partial_bce':'R1','R2_partial_dro':'R2','RN_partial_dro_nochange_pilot':'RN','RD_partial_dro_age_pilot':'RD'}
ESC={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}','<':r'\textless{}','>':r'\textgreater{}','\u00d7':r'$\times$','\u2192':r'$\rightarrow$','\u2013':'--','\u2212':'-'}
def esc(x):return ''.join(ESC.get(c,c) for c in str(x))
def num(x,scale=1):return '--' if x is None or not math.isfinite(float(x)) else f'{float(x)*scale:.4f}'
def count(x):return '--' if x is None else f'{int(x):,}'
T=[]
def para(s):T.extend([esc(s),''])
def section(s):T.append(r'\section*{'+esc(s)+'}')
def table(headers,rows,caption):
 n=len(headers);assert all(len(r)==n for r in rows),caption
 T.extend([r'\begin{longtable}{@{}'+'l'+'r'*(n-1)+r'@{}}',r'\caption{'+esc(caption)+r'}\\',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',r'\endfirsthead',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',r'\endhead',r'\midrule',r'\multicolumn{'+str(n)+r'}{r}{Continued on next page}\\',r'\endfoot',r'\bottomrule',r'\endlastfoot'])
 T.extend(' & '.join(esc(v) for v in r)+r' \\' for r in rows);T.extend([r'\end{longtable}',''])
AXIS_ROWS=[];GROUP_ROWS=[];OVERALL_ROWS=[];CI_ROWS=[]
def collect(model,study,mode,report):
 overall=report['overall'];OVERALL_ROWS.append({'model':model,'study':study,'mode':mode,**overall})
 rows=[]
 for axis in AXES:
  a=report['axes'][axis];ds=a['ding_style_fractions'] or {}
  item={'model':model,'study':study,'mode':mode,'axis':axis,'groups_eligible':len(a['eligible_groups']),'F_FPR_percent':None if ds.get('F_FPR') is None else 100*ds['F_FPR'],'F_DP_percent':None if ds.get('F_DP') is None else 100*ds['F_DP'],'es_AUC_percent':None if ds.get('es_AUC') is None else 100*ds['es_AUC'],'fpr_gap_pp':None if a['fpr_gap'] is None else 100*a['fpr_gap'],'fnr_gap_pp':None if a['fnr_gap'] is None else 100*a['fnr_gap'],'worst_group_AUC_percent':None if a['worst_group_auc'] is None else 100*a['worst_group_auc']}
  AXIS_ROWS.append(item)
  rows.append([NAMES[axis],str(item['groups_eligible']),num(item['F_FPR_percent']),num(item['F_DP_percent']),num(item['es_AUC_percent']),num(item['fpr_gap_pp']),num(item['fnr_gap_pp']),num(item['worst_group_AUC_percent'])])
  for group,g in a['groups'].items():
   GROUP_ROWS.append({'model':model,'study':study,'mode':mode,'axis':axis,'group':group,'eligible':group in a['eligible_groups'],**g})
 return rows
T.extend([r'\documentclass[10pt]{article}',r'\usepackage[a4paper,landscape,margin=0.65in]{geometry}',r'\usepackage[T1]{fontenc}',r'\usepackage{booktabs,longtable,array,amsmath,hyperref}',r'\hypersetup{hidelinks}',r'\setlength{\tabcolsep}{4pt}',r'\renewcommand{\arraystretch}{1.08}',r'\setlength{\LTcapwidth}{\textwidth}',r'\setlength{\LTpre}{8pt}',r'\setlength{\LTpost}{8pt}',r'\begin{document}',r'\raggedbottom',r'\title{Verified Dataset Splits and FairForge-2 Fairness Results}',r'\author{CES-FL research reporting snapshot}',r'\date{2 November 2026}',r'\maketitle',r'\small'])
para('Reporting only: existing JSON metrics and CSV split metadata were inspected. No new training, detector inference, calibration, score inversion or test-based tuning was performed. Image counts below are frame/image rows, not videos or unique people. No result establishes that the current counterfactual method beats Ding.')
section('1. Exact current dataset splits')
rows=[]
for name in ['Ding FF++','FairForge-2 public']:
 c={r['split']:r['rows'] for r in splits['records'] if r['dataset']==name};rows.append([name,count(c['train']),count(c['val']),count(c['test']),count(sum(c.values()))])
table(['Dataset','TRAIN','VAL','TEST','Total'],rows,'Current frozen split sizes; native Ding training/validation/test counts were independently cross-checked.')
table(['Dataset / split','Images','Real','Fake'],[[r['dataset']+' / '+r['split'].upper(),count(r['rows']),count(r['real']),count(r['fake'])] for r in splits['records']],'Class counts in the exact current manifests.')
table(['External dataset','TRAIN used here','VAL used here','TEST','Real','Fake'],[[r['dataset'],'--','--',count(r['rows']),count(r['real']),count(r['fake'])] for r in splits['external_tests']],'Ding-protocol external tests. FF++ supplies training and validation; these domains are test-only in this protocol.')
para('FairForge-2 public is our current release. Historical internal/full builds and a legacy 57,665-image FF2 evaluation subset are different inventories, retained separately below. Exact paths and SHA256 values are in SPLITS.json. Frame membership matching does not establish true-person/donor disjointness; the Ding benchmark has disclosed contributor overlap.')
section('2. FairForge-2 public in-domain validation: current CES-FL pretraining')
o=p['validation']['overall'];cal=p['calibration']
table(['Model / seed','Epochs','Selected epoch','VAL images','AUC (\%)','FPR (\%)','FNR (\%)','Logit threshold'],[['CES-FL P / 17',str(p['completed_epochs']),str(p['selected_epoch']),count(o['n']),num(o['auc'],100),num(o['fpr'],100),num(o['fnr'],100),num(cal['threshold_logit'])]],'Original-only BCE pretraining on public TRAIN; evaluation and selection on public VAL, not TEST.')
para(f"Calibration uses this model's VAL real images only: requested FPR 10%, achieved {cal['achieved_fp']}/{cal['n_validation_real']} = {cal['achieved_fpr']*100:.6f}%. No TEST result exists for this new in-domain model. It used no counterfactual or fairness training loss; the fairness numbers are measured evaluation outcomes, not proof of a fairness improvement.")
headers=['Axis','Eligible groups',r'$F_{\mathrm{FPR}}$ (\%)',r'$F_{\mathrm{DP}}$ (\%)',r'$es$-AUC (\%)','FPR gap (pp)','FNR gap (pp)','Worst AUC (\%)']
table(headers,collect('P17','public_in_domain_VAL','VAL_real_FPR10',p['validation']),'Current in-domain validation fairness across every recorded axis.')
para('F_FPR = 100 times the sum of absolute group FPR deviations from the pooled FPR. F_DP is 100 times the max-min positive-rate gap. FPR/FNR gaps are percentage-point max-min differences. All AUCs and rates are displayed as percentages. F_FPR is not a false-positive rate and its scale depends on the number of groups. Age and skin bins are automated proxies, not newly human-reviewed ground truth. Native ids8 is kept as its own grouping; it is not assumed interchangeable with gender x race. Eligible aggregates require at least20real and20fake frames per group; unknown groups remain visible in detailed support tables.')
section('3. FairForge-2 public TEST: FF++-trained CES-FL transfer results')
para('All five models below were trained on FF++, not FairForge-2. Their ten-epoch downstream runs used the existing shared ten-epoch FF++ detection-pretrained encoder. Seed17. R0=frozen BCE; R1=partial BCE; R2=partial group-DRO; RN=no-change diagnostic; RD=age-edit diagnostic. RN/RD use only six reviewed sources and do not establish broad demographic or forensic-selectivity validity. The frozen global threshold comes from FF++ VAL; it was never calibrated on FairForge-2.')
rows=[]
for arm,x in TRANSFERS.items():
 m=x['metrics']['calibrated'];o=m['overall'];g=m['axes']['gender_race'];ds=g['ding_style_fractions'];rows.append([SHORT[arm],count(o['n']),num(o['auc'],100),num(o['fpr'],100),num(o['fnr'],100),num(ds['F_FPR'],100),num(g['fpr_gap'],100),num(g['worst_group_auc'],100)])
table(['Arm','TEST images','AUC (\%)','FPR (\%)','FNR (\%)',r'G$\times$R $F_{\mathrm{FPR}}$ (\%)',r'G$\times$R FPR gap (pp)',r'G$\times$R worst AUC (\%)'],rows,'Existing public held-out transfer measurements at each stored FF++ validation threshold.')
para('Transfer detection is weak: these AUCs are below50%. Values were not inverted or hidden. They cannot be compared with the99.8105% in-domain VAL score as if they measured the same experiment.')
for mode,title in [('calibrated','Stored FF++ validation threshold'),('fixed_probability_0_5','Fixed probability0.5 (logit0)')]:
 allrows=[]
 for arm,x in TRANSFERS.items():
  rows=collect(SHORT[arm],'FF++_to_public_TEST',mode,x['metrics'][mode]);allrows.extend([[SHORT[arm],*v] for v in rows])
 table(['Arm',*headers],allrows,'All six fairness axes for public TEST transfer: '+title+'.')
section('4. Historical CFES-D: public FairForge-2 in-domain TEST')
para('Historical CLIP ViT-L/14 CFES-D, seed7, is a different model/configuration from current Xception CES-FL. Public TRAIN lineage is recorded. The full TEST report has87,564images; its leakfree-video companion has the same n and AUC and is not counted as another independent experiment. Age/skin-axis exports were not found in these legacy reports. Threshold conventions differ from current CES-FL, so these rows are contextual records, not matched superiority comparisons.')
legacy_rows=[]
for mode,label in [('ding@0.5','Native probability0.5'),('fair@val','Stored VAL threshold')]:
 o=legacy[mode]['overall'];legacy_rows.append([label,num(o['auc'],100),num(o['FPR'],100),num(1-o['TPR'],100),num(legacy['threshold_val']) if mode=='fair@val' else '0.5'])
cm=legacycal['sets']['ff2_test']['calibrated'];legacy_rows.append(['Legacy VAL-intercept calibration',num(cm['AUC']),num(cm['FPR@0.5']),num(100-cm['TPR@0.5']),'0.5 after intercept'])
table(['Stored operating mode','AUC (\%)','FPR (\%)','FNR (\%)','Threshold'],legacy_rows,'Historical public-test CFES-D detection at all located operating modes; FNR is the arithmetic complement of stored TPR.')
legacy_axis=[]
for mode,label in [('ding@0.5','Native0.5'),('fair@val','Stored VAL')]:
 for axis in ['gender','race','intersection']:
  a=legacy[mode][axis];legacy_axis.append([label,axis,num(a['F_FPR']),num(a['F_DP']),num(a['es_AUC']),num(a['fpr_gap']),num(a['tpr_gap']),num(a['worst_group_auc'])])
for axis in ['gender','race','intersection']:
 legacy_axis.append(['VAL-intercept',axis,num(cm[axis+'_F_FPR']),num(cm[axis+'_F_DP']),num(cm[axis+'_es_AUC']),'--','--','--'])
table(['Operating mode','Axis',r'$F_{\mathrm{FPR}}$ (\%)',r'$F_{\mathrm{DP}}$ (\%)',r'$es$-AUC (\%)','FPR gap (pp)','FNR gap (pp)','Worst AUC (\%)'],legacy_axis,'Historical CFES-D fairness exports. Native aggregate entries already use percentage units; they were not multiplied by100 again. Blank entries were not exported.')
para(f"Legacy common-intercept calibration reports b={legacycal['intercept_b']:.8f}, objective: {legacycal['objective']}. This stored calibration was not re-fit here. The operating-point trade-off is visible in FPR and FNR; a tiny F_FPR at a conservative cutoff does not by itself establish strong detection.")
section('5. Archived Ding FF2 references: different evaluation inventories')
para('The archived FF++-trained Ding exports below use either a57,665-row legacy subset or the182,884-row historical internal/full FF2 test. Neither is the current87,564-row public TEST. These native metrics are retained separately. Their original operating/eligibility conventions were not redefined or recalibrated here. Full-stage archive GDA has the documented detached-fairness gradient defect and is not certified as the paper-trained differentiable method. The epoch25 and epoch49 exports are both retained; no test-best checkpoint is selected.')
rows=[]
for item in ding_exports:
 d=item['data']
 for axis in ['gender','race','intersection']:
  a=d.get(axis,{})
  rows.append([item['file'].replace('.json',''),count(d.get('n')),axis,num(d.get('overall_auc')),num(a.get('F_FPR')),num(a.get('F_DP')),num(a.get('es_AUC'))])
table(['Archived export','Eval images','Axis','AUC (\%)',r'$F_{\mathrm{FPR}}$ (\%)',r'$F_{\mathrm{DP}}$ (\%)',r'$es$-AUC (\%)'],rows,'Previously calculated Ding transfer fairness; distinct legacy FF2 inventories, not current public in-domain results.')
rows=[]
for x in ding_evidence['historical_saved_score_recomputation']:
 rows.append(['Ding classification' if 'base_ff2_test' in x['path'] else 'Ding archived SFD+GDA',count(x['rows']),num(x['auc_percent']),'--'])
table(['Historical in-domain model','Internal TEST images','AUC (\%)','Fairness export in reused record'],rows,'Ding trained on the historical larger FF2 build: AUC-only records. No new raw-score evaluation was performed.')
para('That historical internal build records502,035TRAIN /74,974VAL /182,884TEST images. These counts and AUC-only values come from the existing baseline evidence record, not the current public split. A matched Ding model trained and evaluated on the current public split remains unavailable in this inspected record. Do not replace that missing row with internal-build or transfer results.')
section('6. Detailed public demographic group support and errors')
para('The following rows retain every recorded group, including unknown/ineligible groups. AUC is undefined for a group lacking either class. Eligibility is the source report status, not a demographic-truth guarantee. All percentages use the stored global operating point; no group-specific threshold was introduced.')
rows=[]
for g in GROUP_ROWS:
 rows.append([g['model'],g['mode'],NAMES[g['axis']],g['group'],count(g['n_real']),count(g['n_fake']),num(g['auc'],100),num(g['fpr'],100),num(g['fnr'],100),'yes' if g['eligible'] else 'no'])
table(['Model','Operating mode','Axis','Group','Real n','Fake n','AUC (\%)','FPR (\%)','FNR (\%)','Eligible'],rows,'All available group-wise metrics: current public VAL and both stored transfer TEST operating modes. ModelP17 means in-domain VAL; R0/R1/R2/RN/RD mean FF++-trained TEST transfer.')
section('7. Previously calculated transfer confidence intervals')
para('These are the existing1000-replicate percentile intervals over recorded connected-contributor components, conditional on seed17 and frozen thresholds/checkpoints. They do not measure training-seed variability, paired method differences, hidden-donor uncertainty or label uncertainty. Undefined replicates were omitted by the original calculation; valid counts are shown. No confidence intervals were newly calculated for in-domain P17 or historical CFES-D here. The transfer intervals below apply only to the stored validation-calibrated operating point, not the separate fixed0.5 report.')
rows=[]
for arm,x in TRANSFERS.items():
 c=x['confidence_intervals']
 for metric,v in c.get('intervals',{}).items():
  item={'model':SHORT[arm],'metric':metric,'low_percent':100*v['low'],'high_percent':100*v['high'],'valid_replicates':v['valid_replicates'],'requested_replicates':c['replicates'],'recorded_components':c['components']};CI_ROWS.append(item)
  rows.append([SHORT[arm],metric,num(item['low_percent']),num(item['high_percent']),count(v['valid_replicates']),count(c['replicates']),count(c['components'])])
table(['Arm','Metric','95\% lower','95\% upper','Valid reps','Requested','Components'],rows,'All stored transfer uncertainty intervals. AUC/rates are percent; gap intervals are percentage points; F_FPR/F_DP follow the percentage-scaled metric definitions above.')
section('8. Source records and remaining missing results')
para('There are no trained results for the new three-seed CDF/complete-DRO matrix, no full validated manipulation-response training result, and no public TEST score for the current FF2 P17 run. Missing values mean unavailable or undefined, never zero. This export does not authorize training, generation or held-out evaluation. Further FairForge-2 experiments remain held.')
para('The current models, historical models, thresholds, group labels, training sets and evaluation roles differ. Report each block with its own protocol. Do not rank the mixed blocks as an apples-to-apples model comparison. Full machine-readable snapshots, native source hashes, normalized axis/group CSVs and exact split manifests accompany this standalone TeX file.')
T.append(r'\begingroup\footnotesize')
for i,(path,sha) in enumerate(SOURCES.items(),1):
 T.extend([r'\noindent\textbf{Source '+str(i)+r':}\quad\path{'+path+r'}\\',r'\texttt{SHA256: '+sha+r'}\par\medskip'])
T.append(r'\endgroup')
T.append(r'\end{document}')
(ROOT/'ding_and_fairforge2_tables.tex').write_text('\n'.join(T)+'\n')
snapshot={'observed_management_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reporting_repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'source_sha256':SOURCES,'splits':splits,'public_in_domain_validation':p,'public_test_transfer':TRANSFERS,'legacy_CFESD_public':legacy,'legacy_CFESD_calibration':legacycal,'legacy_CFESD_leakfree_summary':{'n':leakfree['n'],'auc':leakfree['auc'],'n_dropped':leakfree['n_dropped']},'Ding_archived_different_inventory_exports':ding_exports,'historical_Ding_internal_AUC_record':ding_evidence,'new_training':False,'new_detector_evaluation':False,'new_calibration':False}
(ROOT/'SNAPSHOT.json').write_text(json.dumps(snapshot,indent=2,allow_nan=False)+'\n')
for name,records in [('AXIS_METRICS.csv',AXIS_ROWS),('GROUP_METRICS.csv',GROUP_ROWS),('OVERALL_METRICS.csv',OVERALL_ROWS),('TRANSFER_EXISTING_CI.csv',CI_ROWS)]:
 fields=list(dict.fromkeys(k for row in records for k in row))
 with (ROOT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
summary={'status':'PASS','tex':str(ROOT/'ding_and_fairforge2_tables.tex'),'source_records':len(SOURCES),'public_axis_rows':len(AXIS_ROWS),'public_group_rows':len(GROUP_ROWS),'existing_CI_rows':len(CI_ROWS),'Ding_legacy_exports':len(ding_exports),'source_metrics_only':True}
(ROOT/'EXPORT_CHECK.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
