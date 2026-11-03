"""CPU-only fixed-VAL paired uncertainty and preregistered decision; no training/TEST."""
import argparse,importlib.util,json,hashlib
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.run);out=Path(a.out);out.mkdir(exist_ok=False)
 s=importlib.util.spec_from_file_location('cesfl',Path(a.source)/'CES_FL/src/ces_fl_forensic_response.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);protocol=read(root/'decision_protocol.json');reference=read(root/'diagnostics/D_validation_scores_PRIVATE.json')
 assert reference['split']=='validation' and reference['test_read'] is False
 records={}
 pairs=[('D_vs_C',root/'diagnostics/D_validation_scores_PRIVATE.json',root/'diagnostics/C_validation_scores_PRIVATE.json')]
 for name in ['E_support','F_support_ramp3']:
  run=root/name/'training';r=read(run/'RUN_VERIFIED.json');scores=read(run/'validation_scores_PRIVATE.json')
  assert (run/'DONE').exists() and r['epochs_completed']==10 and r['source_sha256']==m.sha(m.__file__) and not r['test_evaluated']
  assert r['selected_checkpoint_sha256']==m.sha(run/'best.pt')==scores['checkpoint_sha256'];assert scores['split']=='validation' and scores['test_read'] is False
  assert m.metric_report(scores['rows'],scores['logits'],scores['threshold'],scores['min_support'])==r['validation']
  assert scores['rows']==reference['rows'] and scores['manifest_sha256']==reference['manifest_sha256']
  records[name]=r;pairs.append((name+'_vs_D',run/'validation_scores_PRIVATE.json',root/'diagnostics/D_validation_scores_PRIVATE.json'))
 pairs.append(('F_vs_E',root/'F_support_ramp3/training/validation_scores_PRIVATE.json',root/'E_support/training/validation_scores_PRIVATE.json'))
 for label,first,second in pairs:
  print(json.dumps(dict(event='BOOTSTRAP_STARTED',comparison=label,replicates=protocol['bootstrap']['replicates'],training_seed_variability=False)),flush=True)
  m.compare(first,second,out/(label+'.json'),protocol['bootstrap']['replicates'],17)
 results={}
 for name,r in records.items():
  v=r['validation'];g=v['axes'];interval=read(out/(name+'_vs_D.json'))['intervals']['gender_race/F_FPR'];required=protocol['bootstrap']['required_usable_primary_replicate_fraction']*protocol['bootstrap']['replicates']
  guards={'primary':g['gender_race']['F_FPR']<protocol['primary_F_FPR_max_exclusive'],'auc':v['overall']['auc']>=protocol['auc_min_inclusive'],
   'gender':g['gender']['F_FPR']<=protocol['gender_F_FPR_max_inclusive'],'age':g['age_bin']['F_FPR']<=protocol['age_F_FPR_max_inclusive'],
   'pooled_fake_FNR':v['overall']['fnr']<=protocol['pooled_fake_FNR_max_inclusive'],
   **{'worst_'+axis:g[axis]['worst_group_auc']>=value for axis,value in protocol['worst_group_auc_min_by_axis'].items()}}
  usable=interval['valid_replicates']>=required;ci_below=interval['ci95'] is not None and interval['ci95'][1]<0
  decision='reject' if not (guards['primary'] and guards['auc']) else 'continue' if all(guards.values()) and usable and ci_below else 'revise'
  results[name]=dict(decision=decision,numerical_guards=guards,primary_interval=interval,uncertainty_usable=usable,primary_ci_entirely_below_zero=ci_below,
    fixed_probability_0_5=m.metric_report(read(root/name/'training/validation_scores_PRIVATE.json')['rows'],read(root/name/'training/validation_scores_PRIVATE.json')['logits'],0.,read(root/name/'training/validation_scores_PRIVATE.json')['min_support']),
    selected_alignment_multiplier=m.alignment_multiplier(r['selected_epoch']-1,read(root/name/'config.json')['alignment_warmup_epochs']),
    selected_epoch=r['selected_epoch'],selected_checkpoint_sha256=r['selected_checkpoint_sha256'],validation=v)
 payload=dict(status='VALIDATION_ANALYSIS_COMPLETED',seed=17,source_sha256=m.sha(m.__file__),results=results,training_seed_variability=False,test_read=False,automatic_promotion=False,selection_optimism='Two exploratory candidates and validation selection; conditional bootstrap is not independent TEST confirmation',warnings=['Recorded contributor IDs are not verified biological identities','Sparse subgroup replicates may be undefined','No new Ding computation or FF2 training'])
 m.write_json(out/'DECISIONS.json',payload);print(json.dumps(dict(status='COMPLETED',decisions={k:v['decision'] for k,v in results.items()},test_read=False)),flush=True)
if __name__=='__main__':main()
