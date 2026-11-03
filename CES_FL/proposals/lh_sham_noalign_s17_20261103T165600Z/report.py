"""Replay saved VAL aggregates and compare all metrics, no inference or calibration."""
import csv,json,sys
from pathlib import Path
from checks import m,HERE,ROOT,CONTROL,read
arm=sys.argv[1];assert arm in ('SHAM_FIX','NO_ALIGN')
r=ROOT/arm; t=r/'training'; c=m.load_config(HERE/(arm+'.json'))
receipt=read(t/'RUN_VERIFIED.json');h=read(t/'history.json');control=read(CONTROL/'training/RUN_VERIFIED.json');ch=read(CONTROL/'training/history.json')
assert (t/'DONE').exists() and receipt['status']=='COMPLETED_VERIFIED' and receipt['epochs_completed']==3
assert receipt['source_sha256']==m.sha(m.__file__) and receipt['config_digest']==m.digest(c) and receipt['test_evaluated'] is False
assert receipt['selected_checkpoint_sha256']==m.sha(t/'best.pt') and receipt['selected_epoch']==max(h,key=lambda e:e['validation']['overall']['auc'])['epoch']
assert control['selected_checkpoint_sha256']==m.sha(CONTROL/'training/best.pt')=='59143159eea223d97bc436b510f13ca90b750f00851071a860f91ec9231d4a06'
exposure=['steps','unique_originals','originals_once_each','original_real_pairs','auxiliary_schedule','auxiliary_planned_draws','auxiliary_forwarded_originals','auxiliary_unique_planned_sources','original_real_pairs_by_axis_group','auxiliary_original_draws_by_axis_group_label','auxiliary_real_pair_draws_by_axis_group','alignment_multiplier']
for a,b in zip(h,ch):
 assert all(a[k]==b[k] for k in exposure)
scores=read(t/'validation_scores_PRIVATE.json')
assert scores['split']=='validation' and scores['test_read'] is False and scores['checkpoint_sha256']==receipt['selected_checkpoint_sha256'] and scores['manifest_sha256']==c['val_sha256']
assert m.metric_report(scores['rows'],scores['logits'],scores['threshold'],scores['min_support'])==receipt['validation']
out=m.fresh_dir(r/'comparison')
stages=[('control_epoch'+str(e['epoch']),e['validation']) for e in ch]+[(arm+'_epoch'+str(e['epoch']),e['validation']) for e in h]+[('control_selected',control['validation']),(arm+'_selected',receipt['validation'])]
lines=['# '+arm+' versus completed lower-LR control','','Validation only; earliest maximum AUC selection, same calibration; F_FPR in fractions.','', '|Run|AUC %|FPR %|FNR %|Gender|Race|Gender×race|Age|Skin|ids8|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
groups=[]; axes=[]
for name,v in stages:
 o=v['overall'];lines.append('|'+name+'|'+'|'.join(f'{100*o[k]:.6f}' for k in ('auc','fpr','fnr'))+'|'+'|'.join(f"{v['axes'][ax]['F_FPR']:.6f}" for ax in ('gender','race','gender_race','age_bin','skin_bin','ids8'))+'|')
 for ax,d in v['axes'].items():
  axes.append(dict(run=name,axis=ax,**{k:d[k] for k in ('F_FPR','F_DP','es_AUC','worst_group_auc','fpr_gap','fnr_gap','auc_gap')}))
  for g,values in d['groups'].items():groups.append(dict(run=name,axis=ax,group=g,eligible=g in d['eligible_groups'],**values))
for fname,rows in [('ALL_FAIRNESS_METRICS.csv',axes),('GROUP_COUNTS_AND_RATES.csv',groups)]:
 with (out/fname).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for ax in m.AXES:
 lines+=['','## '+ax,'','|Run|F_DP|es-AUC %|Worst AUC %|FPR gap pp|FNR gap pp|AUC gap pp|','|---|---:|---:|---:|---:|---:|---:|']
 for name,v in stages:
  d=v['axes'][ax];lines.append('|'+name+'|'+f"{d['F_DP']:.6f}|"+'|'.join(f'{100*d[k]:.6f}' for k in ('es_AUC','worst_group_auc','fpr_gap','fnr_gap','auc_gap'))+'|')
lines+=['','Single-seed validation development; no new uncertainty/significance or TEST claim. All group counts and rates are in the CSV. Existing demographic-label, sparse-support and declared contributor-overlap limitations persist. No automatic extension or combined-change run.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
m.write_json(out/'COMPARISON.json',dict(status='PASS',config=c,source_sha256=m.sha(m.__file__),run=receipt,history=h,control=control,control_history=ch,exposure_matches=True,metrics_replay=True,no_test_access=True))
print('COMPARISON_PASS',arm,flush=True)
