"""Presentation from archived published numeric facts and verified aggregate scores.

No checkpoint/image/manifest reads, model inference, calibration or selection.
Writes new outputs only; never overwrites an earlier table export.
"""
import argparse,csv,hashlib,json,math
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESULTS=HERE.parent
AXES=('gender','race','gender_race')
BLOCKS=('Gender','Race','Intersection')
DATASETS={'DFDC':'dfdc','Celeb-DF':'celebdf','DFD':'dfd'}

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def vector(report,axis=None):
 vals=[report['axes'][a][k] for a in ([axis] if axis else AXES) for k in ('F_FPR','F_DP','es_AUC')]
 return [100*v if v is not None else None for v in vals+[report['overall']['auc']]]
def fmt(v):return 'undefined' if v is None else f'{v:.3f}'
def verify(report):
 for axis in AXES:
  a=report['axes'][axis];groups=[a['groups'][g] for g in a['eligible_groups']]
  expected={'F_FPR':sum(abs(g['fpr']-report['overall']['fpr']) for g in groups),
   'F_DP':max(g['positive_rate'] for g in groups)-min(g['positive_rate'] for g in groups),
   'es_AUC':report['overall']['auc']/(1+sum(abs(g['auc']-report['overall']['auc']) for g in groups))}
  for k,v in expected.items():assert math.isclose(v,a[k],rel_tol=0,abs_tol=1e-12),(axis,k)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
 if out.exists():raise ValueError('Preserve existing table export: '+str(out))
 aggregate=read(HERE/'AGGREGATE.json');paper=read(RESULTS/'ding_format_tables_s17_20261102/source_snapshot.json')['paper']
 assert aggregate['analysis_done']['status']=='PASS' and len(aggregate['analysis'])==12
 assert sum(map(len,aggregate['results'].values()))==8 and not aggregate['test_threshold_search']
 for arm in ('L_A','L_H'):
  for result in aggregate['results'][arm].values():
   for key in ('fixed_probability_0_5','calibrated'):verify(result[key])
 out.mkdir(parents=True)
 note=('All displayed values are100 times native fractions. F_FPR is summed disparity, not a probability. '
  'Published baseline rows are reference results transcribed from Ding et al., arXiv2511.10150v5, Tables1/2. '
  'Our Xception rows are seed17, one original validation-selected checkpoint per arm, ten downstream epochs; '
  'L_A selected epoch3, L_H epoch1. Stage budgets, preprocessing and checkpoint/support policies differ from the paper. '
  'No new Ding computation, inference, calibration or model selection was performed to create these tables. '
  'Same layout does not establish protocol equivalence. No bold best-cell ranking or pooled win count is used. '
  'Paper: https://arxiv.org/html/2511.10150v5. [Uncertainty and support](../UNCERTAINTY.md).')
 all_rows={};tex=['% Requires booktabs and graphicx. Values are100 x native fractions; distinct threshold views must not be combined.']
 for view,key in [('FIXED05','fixed_probability_0_5'),('SOURCE_VAL10','calibrated')]:
  policy=('Our threshold is fixed probability0.5 (logit0); Ding values retain the paper operating protocol.' if view=='FIXED05' else
   'Our threshold is each selected model\'s frozen FF++validation-real10%FPR threshold, applied unchanged to every TEST dataset; Ding values retain the paper operating protocol. Requested validation FPR is not guaranteed TEST FPR.')
  for table in (1,2):
   rows=[]
   for block in (BLOCKS if table==1 else DATASETS):
    published=[r for r in paper['tables'][str(table)] if r['block']==block]
    assert len(published)==7
    rows.extend(published)
    ds='ffpp' if table==1 else DATASETS[block];axis=AXES[BLOCKS.index(block)] if table==1 else None
    for arm in ('L_A','L_H'):
     rows.append(dict(block=block,method='CES-FL '+arm+' [seed17]',backbone='Xception',values_pct=vector(aggregate['results'][arm][ds][key],axis),kind='measured_test'))
   assert len(rows)==27
   all_rows[f'{table}_{view}']=rows
   headers=['Attribute' if table==1 else 'Dataset','Method','Backbone']
   headers+=['F_FPR ↓','F_DP ↓','es-AUC ↑','AUC ↑'] if table==1 else [g+' '+k for g in BLOCKS for k in ('F_FPR ↓','F_DP ↓','es-AUC ↑')]+['AUC ↑']
   content=[f'# Table {table}: '+('FF++ intra-domain' if table==1 else 'Cross-domain')+' — '+view,'',note,'',policy,'',
    '| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]
   for r in rows:content.append('| '+' | '.join([r['block'],r['method'],r['backbone']]+[fmt(v) for v in r['values_pct']])+' |')
   (out/f'TABLE_{table}_{view}.md').write_text('\n'.join(content)+'\n')
   with (out/f'TABLE_{table}_{view}.csv').open('x',newline='') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(headers)
    for r in rows:w.writerow([r['block'],r['method'],r['backbone']]+r['values_pct'])
   def esc(v):return str(v).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('↓',r'$\downarrow$').replace('↑',r'$\uparrow$')
   tex+=['\\begin{table*}[t]','\\centering',f'\\caption{{Table {table}: {esc(view)}. Published references and CES-FL seed17; protocols differ.}}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lll'+'r'*(len(headers)-3)+'}',r'\toprule',' & '.join(map(esc,headers))+r' \\',r'\midrule']
   for r in rows:tex.append(' & '.join(map(esc,[r['block'],r['method'],r['backbone']]+[fmt(v) for v in r['values_pct']]))+r' \\')
   tex += [r'\bottomrule',r'\end{tabular}}',r'\end{table*}','']
 (out/'TABLES_1_2.tex').write_text('\n'.join(tex).rstrip()+'\n')
 lines=['# Complete published baseline rows plus CES-FL L_A/L_H','',note,'',
  '[Table1 fixed0.5](TABLE_1_FIXED05.md) · [Table2 fixed0.5](TABLE_2_FIXED05.md) · [Table1 source-VAL10%](TABLE_1_SOURCE_VAL10.md) · [Table2 source-VAL10%](TABLE_2_SOURCE_VAL10.md) · [LaTeX](TABLES_1_2.tex)','',
  'Every table includes all seven published baseline/method rows in each block followed by L_A and L_H. Historical local Ding experiments and unimplemented rows are deliberately left in the original separate report; these files compare published results with the current two models.','',
  'At fixed0.5 on FF++: L_A FPR9.365%, FNR3.878%; L_H FPR4.319%, FNR7.070%. At source-validation10%: L_A TEST FPR11.305%, L_H10.688%. A fairness change across unequal operating rates must be considered with FNR and utility.','',
  'Primary joint-group uncertainty uses990/1000 usable replicates for FF++,608 for DFDC,587 for Celeb-DF,957 for DFD; retain900minimum. All report intervals are conditional on the single trained seed. DFDC/Celeb-DF joint estimates fail that support requirement. The paper training protocol and ours are not fully matched.','',
  'No post-TEST tuning or selective threshold choice is authorized by this table request.']
 (out/'README.md').write_text('\n'.join(lines)+'\n')
 receipt={'status':'PASS','presentation_only':True,'aggregate_sha256':sha(HERE/'AGGREGATE.json'),'paper_snapshot_sha256':sha(RESULTS/'ding_format_tables_s17_20261102/source_snapshot.json'),'source_commit':aggregate['source_commit'],'rows_per_table':27,'tables':4,'new_inference':False,'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
 (out/'VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
