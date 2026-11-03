"""Render fixed published/historical/new rows, without model computation."""
import argparse,csv,datetime,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-lh-tables-ee73f2f')
AXES=('gender','race','gender_race')
NAMES={'ffpp':'FF++','dfdc':'DFDC','celebdf':'Celeb-DF','dfd':'DFD'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def vec(v,axis=None):
 values=[]
 for ax in ([axis] if axis else AXES):values.extend(v['axes'][ax][k] for k in ('F_FPR','F_DP','es_AUC'))
 values.append(v['overall']['auc'])
 return [100*x if x is not None else None for x in values]
def fmt(x):return 'PENDING' if x is None else f'{x:.3f}'
def md(rows,table):
 headers=['Attribute' if table==1 else 'Dataset','Method / protocol','Backbone']
 headers+=['F_FPR ↓','F_DP ↓','es-AUC ↑','AUC ↑'] if table==1 else [f'{g} {k}' for g in ('Gender','Race','Joint') for k in ('F_FPR ↓','F_DP ↓','es-AUC ↑')]+['Overall AUC ↑']
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]+['| '+' | '.join([r['block'],r['method'],r['backbone']]+[fmt(v) for v in r['values_pct']])+' |' for r in rows])+'\n'
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 plan=read(ROOT/'LOCKED_PLAN.json');paper=read(SOURCE/'CES_FL/results/ding_format_tables_s17_20261102/source_snapshot.json')['paper'];hist=read(SOURCE/'CES_FL/results/ding_reproduced_vs_cesfl_20261102/comparison.json')
 assert sha(SOURCE/'CES_FL/src/ces_fl_head_ablation.py')==plan['source_sha256']
 assert sha(SOURCE/'CES_FL/scripts/lh_locked_benchmarks.py')==plan['runner_sha256']
 for arm in plan['arms'].values():
  assert sha(arm['checkpoint'])==arm['checkpoint_sha256'] and sha(arm['validation_cache'])==arm['validation_cache_sha256']
 for ds in plan['datasets'].values():
  assert sha(ds['path'])==ds['sha256'] and sha(ds['overlap_file'])==ds['overlap_sha256']
 results={};aggregate={'observed_management_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'ee73f2fdf5218e440852917e90281818bbe711bb','model_source_sha256':plan['source_sha256'],'runner_sha256':plan['runner_sha256'],'plan_sha256':sha(ROOT/'LOCKED_PLAN.json'),'models':plan['arms'],'datasets':plan['datasets'],'test_threshold_search':False,'paper_reference':{k:v for k,v in paper.items() if k!='tables'},'results':{},'analysis':{}}
 for arm in ('L_A','L_H'):
  results[arm]={}
  for ds in NAMES:
   r=ROOT/'inference'/arm/ds
   if not (r/'INFERENCE_VERIFIED.json').exists():continue
   receipt=read(r/'INFERENCE_VERIFIED.json');assert receipt['status']=='PASS' and receipt['checkpoint_sha256']==plan['arms'][arm]['checkpoint_sha256'] and receipt['manifest_sha256']==plan['datasets'][ds]['sha256'] and receipt['plan_sha256']==sha(ROOT/'LOCKED_PLAN.json') and receipt['metrics_sha256']==sha(r/'metrics.json') and receipt['scores_sha256']==sha(r/'scores_PRIVATE.json')
   results[arm][ds]=read(r/'metrics.json')
 aggregate['results']=results
 for file in (ROOT/'analysis').glob('*.json'):
  aggregate['analysis'][file.stem]=read(file)
 if (ROOT/'ALLOCATION_END.json').exists():aggregate['allocation_end']=read(ROOT/'ALLOCATION_END.json')
 if (ROOT/'ANALYSIS_DONE.json').exists():
  aggregate['analysis_done']=read(ROOT/'ANALYSIS_DONE.json')
  assert aggregate['analysis_done']['status']=='PASS' and len(aggregate['analysis'])==12
 tables={}
 for view,key in [('FIXED05','fixed_probability_0_5'),('SOURCE_VAL10','calibrated')]:
  tables[view]={}
  for t in (1,2,3):
   blocks=['Gender','Race','Intersection'] if t==1 else (['DFDC','Celeb-DF','DFD'] if t==2 else ['FF++','DFDC','Celeb-DF','DFD'])
   rows=[]
   for block in blocks:
    ds='ffpp' if t==1 else next(k for k,v in NAMES.items() if v==block)
    axis=AXES[blocks.index(block)] if t==1 else None
    rows.extend({**r,'method':'Ding published reference [archived v5; own protocol]'} for r in paper['tables'][str(t)] if r['method']=='Ding et al. (published)' and r['block']==block)
    if t!=3:
     if view=='FIXED05':
      for old in hist['rows']:
       if old['dataset']!=NAMES[ds] or not old['method'].startswith('Ding'):continue
       vals=old['values_pct'];values=vals[AXES.index(axis)*3:AXES.index(axis)*3+3]+[vals[-1]] if axis else vals
       label='Ding classification historical, s17' if 'classification' in old['method'] else 'Ding SFD+GDA historical, s7 [detached fairness]'
       rows.append(dict(block=block,method=label,backbone='Xception',values_pct=values,kind='historical'))
     rows.append(dict(block=block,method='Ding connected-gradient 10epoch adaptation [not implemented]',backbone='Xception',values_pct=[None]*(4 if t==1 else 10),kind='pending'))
     for arm in ('L_A','L_H'):
      report=results[arm].get(ds)
      rows.append(dict(block=block,method=f'{arm}, s17, selected ep{plan["arms"][arm]["selected_epoch"]}',backbone='Xception',values_pct=vec(report[key],axis) if report else [None]*(4 if t==1 else 10),kind='new_fixed_model_test' if report else 'pending'))
    else:
     rows.append(dict(block=block,method='Current L_H ResNet-50 [implementation/P50/A/H training pending]',backbone='ResNet-50',values_pct=[None]*10,kind='pending'))
   tables[view][str(t)]=rows
   note=('Published rows retain their archived paper values and own protocol. Historical rows are fixed0.5 and are included only in FIXED05. New rows preserve one selected model per arm; no per-cell selection. All values are100×native fractions; F_FPR is a sum, not a probability. '
     +'New rows use fixed probability0.5.' if view=='FIXED05' else 'Published rows are reference values under their own protocol. New rows use each selected checkpoint\'s global source FF++ validation10%FPR threshold on every target dataset. Target TEST scores/labels never recalibrate it. Historical fixed0.5 rows are omitted here. All displayed values are100×native fractions; no protocol-equivalent win is inferred.')
   text=f'# Table {t}: '+('FF++' if t==1 else 'Cross-domain' if t==2 else 'ResNet-50')+f' — {view}\n\n'+note+'\n\n'+md(rows,t)
   if t==3:text+='\nNo new ResNet L_H computation has completed. Historical ResNet A/B/C cannot fill this row.\n'
   (out/f'TABLE_{t}_{view}.md').write_text(text)
 aggregate['table_rows']=tables
 (out/'AGGREGATE.json').write_text(json.dumps(aggregate,indent=2)+'\n')
 pergroup=[]
 for arm,datasets in results.items():
  for ds,r in datasets.items():
   for view,key in [('SOURCE_VAL10','calibrated'),('FIXED05','fixed_probability_0_5')]:
    for axis,ad in r[key]['axes'].items():
     for g,v in ad['groups'].items():pergroup.append(dict(model=arm,dataset=ds,view=view,axis=axis,group=g,eligible=g in ad['eligible_groups'],**v))
 if pergroup:
  with (out/'GROUP_COUNTS_AND_METRICS.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(pergroup[0]),lineterminator='\n');w.writeheader();w.writerows(pergroup)
 def esc(v):return str(v).replace('\\','\\textbackslash{}').replace('_','\\_').replace('%','\\%').replace('&','\\&').replace('#','\\#')
 tex=['% Requires graphicx and booktabs; all numbers are percentages; pending cells are not measured scores.']
 for view,sets in tables.items():
  for t,rows in sets.items():
   t=int(t);headers=['Attribute' if t==1 else 'Dataset','Method','Backbone']+(['F-FPR','F-DP','es-AUC','AUC'] if t==1 else [g+' '+m for g in ['Gender','Race','Joint'] for m in ['F-FPR','F-DP','es-AUC']]+['AUC'])
   tex+=['\\begin{table*}[t]','\\centering',f'\\caption{{Table {t}, {esc(view)}. Protocol differences and pending rows are described in the accompanying Markdown.}}','\\resizebox{\\textwidth}{!}{%','\\begin{tabular}{lll'+'r'*(len(headers)-3)+'}', '\\toprule',' & '.join(map(esc,headers))+' \\\\', '\\midrule']
   for row in rows:tex.append(' & '.join(map(esc,[row['block'],row['method'],row['backbone']]+[fmt(v) for v in row['values_pct']]))+' \\\\')
   tex+=['\\bottomrule','\\end{tabular}}','\\end{table*}','']
 (out/'TABLES_1_2_3.tex').write_text('\n'.join(tex).rstrip()+'\n')
 done=sum(len(x) for x in results.values());ci=bool(aggregate.get('analysis_done'))
 lines=['# Locked L_A/L_H Tables1–3','',aggregate['observed_management_utc']+' management UTC.','',f'Verified TEST inference: {done}/8. Bootstrap analysis: '+('COMPLETED.' if ci else 'PENDING; point estimates only until the CPU job finishes.'),'',
  'Exact operator commit `ee73f2fdf5218e440852917e90281818bbe711bb`; historical model source `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`. L_A epoch3 checkpoint `aee9670ca011937d457846fbf13717445c5607715064dd6a54da5a82bda580ae`; L_H epoch1 checkpoint `6ef2caad3979d101fd94d7350aa3c1a50f95ee8307673cfaae63ac9d89af36f0`. Both ten-epoch runs completed/native replay PASS.','',
  '[Findings and limitations](FINDINGS.md) · [Table1 fixed0.5](TABLE_1_FIXED05.md) · [Table2 fixed0.5](TABLE_2_FIXED05.md) · [Table3 pending ResNet](TABLE_3_FIXED05.md) · [Table1 source-VAL10%](TABLE_1_SOURCE_VAL10.md) · [Table2 source-VAL10%](TABLE_2_SOURCE_VAL10.md) · [TeX](TABLES_1_2_3.tex) · [Paired intervals and support](UNCERTAINTY.md) · [All interval cells](BOOTSTRAP_INTERVALS.csv).','',
  'Published Ding values are preserved archived-v5 transcriptions from the pinned source snapshot. Historical classification and GDA rows retain their original seeds/settings and fixed0.5 protocol; the archived GDA fairness term was detached. No new Ding10 result exists yet. The prepared connected-gradient10epoch adaptation, ResNet L_H and prospective cosine schedule require implementations/checks and assigned budgets. The unrelated matched50+50 recipe remains deferred.','',
  'Every L_A/L_H row uses the original validation-selected checkpoint and unchanged threshold. FF++ fit:TEST contributor declarations are unions of the existing six-key audit; cross-domain checks apply strict zero recorded-contributor/content overlap to the fixed historical manifests, with no new exemptions. Biological identity independence remains unverified. DFDC retains1,476native duplicate rows; component bootstrap handles recorded dependence but cannot prove independent people. Age/skin proxy-label limits remain.','',
  'Final analysis reports1000 connected-contributor draws, fixed eligibility and actual usable counts, with the900minimum retained. H-minus-A is paired within each dataset and threshold view. Insufficient support or intervals crossing zero do not establish equivalence or a win. One trained seed does not measure training-seed variability. No test-driven model, threshold or training-design changes are allowed.','',
  'Tables show format-aligned comparisons, not fully matched paper protocols. Prior project TEST exposure is disclosed; these checkpoints were frozen before this one authorized evaluation. Original FF++ budget funds GPU inference; no FairForge-2 baseline hours were transferred. Private images, row scores and checkpoints stay on cluster.']
 (out/'STATUS.md').write_text('\n'.join(lines)+'\n')
 print('LOCKED_TABLES_RENDERED',done,'verified inference outputs','bootstrap_done',ci,flush=True)
if __name__=='__main__':main()
