"""Fixed post-inference CPU analysis. No inference, training or model selection."""
import concurrent.futures,importlib.util,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-lh-tables-ee73f2f')
def model():
 s=importlib.util.spec_from_file_location('locked',SOURCE/'CES_FL/scripts/lh_locked_benchmarks.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def single(task):
 arm,ds=task;m=model();m.torch.set_num_threads(1)
 out=ROOT/'analysis'/(arm+'_'+ds+'.json')
 m.analyze(ROOT/'LOCKED_PLAN.json',arm,ds,ROOT/'inference'/arm/ds,out)
 return str(out)
def pair(ds):
 m=model();m.torch.set_num_threads(1)
 paths={a:ROOT/'inference'/a/ds/'scores_PRIVATE.json' for a in m.ARMS}
 scores={a:json.loads(p.read_text()) for a,p in paths.items()};a,b=scores['L_H'],scores['L_A']
 assert a['rows']==b['rows'] and a['manifest_sha256']==b['manifest_sha256'] and a['min_support']==b['min_support']
 result={}
 for key,ta,tb in [('native_selected_VAL_FPR10',a['threshold'],b['threshold']),('fixed_probability_0.5',0.,0.)]:
  result[key]=m.core.bootstrap(a['rows'],a['logits'],ta,a['min_support'],1000,17,other=(b['logits'],tb))
 result.update(dataset=ds,paired_difference='L_H minus L_A',native_scores_unchanged=True,score_hashes={a:m.core.sha(p) for a,p in paths.items()},required_usable_draws=900,training_seed_count=1,calibration_changed=False)
 out=ROOT/'analysis'/('H_minus_A_'+ds+'.json');assert not out.exists();m.core.write_json(out,result);return str(out)
def main():
 m=model();end=json.loads((ROOT/'ALLOCATION_END.json').read_text());assert end['status']=='COMPLETED' and len(end['completed'])==8
 out=ROOT/'analysis';out.mkdir(exist_ok=False)
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(single,(a,d)) for a in m.ARMS for d in m.DATASETS]
  for future in concurrent.futures.as_completed(futures):print('ANALYSIS_PASS',future.result(),flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for result in pool.map(pair,m.DATASETS):print('PAIRED_ANALYSIS_PASS',result,flush=True)
 m.core.write_json(ROOT/'ANALYSIS_DONE.json',dict(status='PASS',single_model_analyses=8,paired_dataset_analyses=4,bootstrap_draws=1000,seed=17,source_commit='ee73f2fdf5218e440852917e90281818bbe711bb',test_threshold_search=False,no_additional_inference=True))
if __name__=='__main__':main()
