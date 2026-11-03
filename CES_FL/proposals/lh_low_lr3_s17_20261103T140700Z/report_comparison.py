"""Complete VAL-only comparison, no inference, training or threshold selection."""
import argparse,csv,datetime,importlib.util,json
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parents[2]
spec=importlib.util.spec_from_file_location('plain',SOURCE/'CES_FL/src/ces_fl_head_ablation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/lh_low_lr3_s17_20261103T140700Z')
OLD=Path('/home/sachin.chaudhary/ces_fl_runs/plain_head_s17_20261103T101418Z/L_H')
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=m.fresh_dir(a.out)
 c=m.load_config(HERE/'config.json');old=read(OLD/'config.json')
 assert {k for k in c if old[k]!=c[k]}=={'head_lr','encoder_lr','epochs'}
 root=ROOT/'training';r=read(root/'RUN_VERIFIED.json');h=read(root/'history.json');oh=read(OLD/'history.json');oref=read(OLD/'RUN_VERIFIED.json')
 assert (root/'DONE').exists() and r['status']=='COMPLETED_VERIFIED' and r['epochs_completed']==3 and r['config_digest']==m.digest(c) and r['source_sha256']==m.sha(m.__file__) and not r['test_evaluated']
 assert [e['epoch'] for e in h]==[1,2,3]
 assert r['selected_epoch']==max(h,key=lambda x:x['validation']['overall']['auc'])['epoch']
 assert r['selected_checkpoint_sha256']==m.sha(root/'best.pt')
 assert len(oh)==10 and m.sha(OLD/'best.pt')==oref['selected_checkpoint_sha256']
 ck=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
 assert ck['validation']==r['validation'] and ck['config']==c
 assert ck['calibration']['threshold_logit']==r['validation']['threshold_logit']
 exposure=['steps','unique_originals','originals_once_each','original_real_pairs','auxiliary_schedule','auxiliary_planned_draws','auxiliary_forwarded_originals','auxiliary_unique_planned_sources','original_real_pairs_by_axis_group','auxiliary_original_draws_by_axis_group_label','auxiliary_real_pair_draws_by_axis_group','alignment_multiplier']
 for new,previous in zip(h,oh):
  assert all(new[k]==previous[k] for k in exposure),new['epoch']
 scores=read(root/'validation_scores_PRIVATE.json')
 assert scores['split']=='validation' and scores['test_read'] is False and scores['checkpoint_sha256']==r['selected_checkpoint_sha256'] and scores['manifest_sha256']==c['val_sha256']
 assert m.metric_report(scores['rows'],scores['logits'],scores['threshold'],scores['min_support'])==r['validation']
 fixed=m.metric_report(scores['rows'],scores['logits'],0.,scores['min_support'])
 stages=[('original_L_H_epoch'+str(e['epoch']),e['epoch'],e['validation']) for e in oh[:3]]
 stages += [('original_retained_best_after10',oref['selected_epoch'],oref['validation'])]
 stages += [('low_lr_epoch'+str(e['epoch']),e['epoch'],e['validation']) for e in h]
 stages += [('low_lr_selected_after3',r['selected_epoch'],r['validation'])]
 lines=['# L_H lower-learning-rate three-epoch validation experiment','',datetime.datetime.now(datetime.timezone.utc).isoformat(),'','All new scientific computation uses TRAIN/VAL only. Exactly three config fields changed. Reference models/histories/checkpoints are preserved.','',
 '|Model/epoch|AUC %|Achieved FPR %|FNR %|Gender F_FPR|Race F_FPR|Joint F_FPR|Age F_FPR|Skin F_FPR|ids8 F_FPR|',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,ep,v in stages:
  axes='|'.join(f"{v['axes'][ax]['F_FPR']:.6f}" for ax in ('gender','race','gender_race','age_bin','skin_bin','ids8'))
  o=v['overall'];lines.append(f"|{name}|{100*o['auc']:.6f}|{100*o['fpr']:.6f}|{100*o['fnr']:.6f}|{axes}|")
 pergroup=[];axis_rows=[]
 for name,ep,v in stages+[('low_lr_selected_fixed0.5',r['selected_epoch'],fixed)]:
  for ax,d in v['axes'].items():
   axis_rows.append(dict(model=name,epoch=ep,axis=ax,**{k:d[k] for k in ('F_FPR','F_DP','es_AUC','worst_group_auc','fpr_gap','fnr_gap','auc_gap')}))
   for group,g in d['groups'].items():pergroup.append(dict(model=name,epoch=ep,axis=ax,group=group,eligible=group in d['eligible_groups'],**g))
 for filename,rows in [('ALL_FAIRNESS_METRICS.csv',axis_rows),('GROUP_COUNTS_AND_RATES.csv',pergroup)]:
  with (out/filename).open('x',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 for ax in m.AXES:
  lines+=['','## '+ax,'','|Model|F_DP|es-AUC %|Worst AUC %|FPR gap pp|FNR gap pp|AUC gap pp|','|---|---:|---:|---:|---:|---:|---:|']
  for name,ep,v in stages:
   d=v['axes'][ax];lines.append('|'+name+'|'+f"{d['F_DP']:.6f}|"+'|'.join(f'{100*d[k]:.6f}' for k in ('es_AUC','worst_group_auc','fpr_gap','fnr_gap','auc_gap'))+'|')
 allocation=read(ROOT/'ALLOCATION_END.json') if (ROOT/'ALLOCATION_END.json').exists() else None
 result=dict(status='COMPLETED_VERIFIED',source=read(ROOT/'SOURCE_CHECKOUT.json'),config=c,config_sha256=m.sha(HERE/'config.json'),run=r,history=h,original_epochs1to3=oh[:3],original_retained_best=oref,comparison_stages=[dict(name=n,epoch=ep,validation=v) for n,ep,v in stages],selected_fixed_probability_0_5=fixed,
  exposure_matches_original_first3=True,train_drift=read(ROOT/'train_drift/TRAIN_DRIFT.json'),allocation=allocation,no_test_access=True,
  limitations=['One seed; validation development after prior broader project TEST exposure.','No new TEST inference/tuning authorized.','Proxy demographic labels and163 declaredTRAIN/VALcontributor overlaps remain.','No new uncertainty analysis performed here; no significance/seed stability claim.','Original10epochbest includes greater training exposure; matched epochs1–3 shown separately.'])
 lines+=['','The actual loss is unchanged. Lower transformed sham differences alone do not establish lower raw differences or causal improvement. No TEST result or best fairness checkpoint is substituted.','',
 f"Selected new checkpoint epoch{r['selected_epoch']}, SHA `{r['selected_checkpoint_sha256']}`. Source/config/commands/provenance, complete histories and runtime are in COMPARISON.json; all group support/rates and every metric are in the CSV exports.",'',
 'Single-seed validation development only. No significance conclusion or overall fairness win follows without supported uncertainty. Every regression is retained in the same tables.']
 m.write_json(out/'COMPARISON.json',result);(out/'REPORT.md').write_text('\n'.join(lines)+'\n');print('VAL_COMPARISON_PASS',out)
if __name__=='__main__':main()
