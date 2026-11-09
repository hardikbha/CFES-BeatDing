"""Independent table/selection checks; optional verification against official paper HTML."""
import argparse
import csv
import hashlib
import json
import math
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'SNAPSHOT.json').read_text())
with (ROOT/'BEST_CELL_PROVENANCE.csv').open() as f:ledger=list(csv.DictReader(f))
OWN={'CES-FL P','CES-FL A','CES-FL B','L_A (plain-head BCE)','L_H-PS (ours)','RF-S (ours)','RF-SJg (ours)'}
DS={'FF++':'ffpp','DFDC':'dfdc','Celeb-DF':'celebdf','DFD':'dfd'}
checks=0
for cell in ledger:
    n=int(cell['table']);m=cell['metric']
    if n<4:
        ds='ffpp' if n==1 else DS[cell['block']]
        rows=[r for r in s['benchmark_candidates'] if r['dataset']==ds and r['backbone']==('ResNet-50' if n==3 else 'Xception')]
        if m=='overall AUC':vals=[(r,r['overall_auc_pct']) for r in rows];lower=False
        else:
            ax,key=m.split(' ',1);vals=[(r,r['axes'][ax][key]) for r in rows];lower=key!='es_AUC_pct'
    else:
        prefix,ax,key=m.rsplit(' ',2)
        rows=s['ff2_validation' if n==5 else 'ff2_test']
        rows=[r for r in rows if r['axis']==('gender' if ax=='None' else ax)
              and (r['model'] in OWN if prefix=='Best own' else r['model']==prefix) and r.get(key) is not None]
        short={'BCE':'BCE','DAG-FDD':'DAG','DAW-FDD':'DAW','Ding corrected GDA':'GDA','CES-FL P':'P',
               'CES-FL A':'A','CES-FL B':'B','L_A (plain-head BCE)':'LA','L_H-PS (ours)':'LH','RF-S (ours)':'S','RF-SJg (ours)':'J'}
        vals=[(dict(r,tag=short[r['model']]+'-'+str(r['seed'])),r[key]) for r in rows]
        lower=not (key in ('overall_auc_pct','es_AUC_pct','worst_group_auc_pct'))
    target=(min if lower else max)(v for _,v in vals)
    assert math.isclose(float(cell['value']),target,rel_tol=0,abs_tol=1e-12),cell
    named=[r for r,v in vals if r['tag']==cell['source_tag'] and abs(v-target)<1e-12]
    assert named,cell
    assert int(cell['seed'])==named[0]['seed']
    for metric in ['auc','fpr','fnr']:
        assert math.isclose(float(cell['source_run_'+metric.upper()+'_pct']),named[0]['overall_'+metric+'_pct'],abs_tol=1e-12)
    checks+=1

assert [len(s['paper']['tables'][str(n)]) for n in [1,2,3]]==[21,21,20]
tex=(ROOT/'SIX_TABLES_BEST_OBSERVED.tex').read_text()
assert re.findall(r'\\section\*\{Table ([1-6])\.',tex)==list('123456')
assert len(s['benchmark_candidates'])==68
assert len({(r['tag'],r['dataset']) for r in s['benchmark_candidates']})==68
assert s['ff2_split_totals']['TRAIN']=={'real':16777,'fake':201489}
assert s['ff2_split_totals']['SELECT']=={'real':2438,'fake':28669}
assert s['ff2_split_totals']['TEST']=={'real':5441,'fake':82123}
assert len({r['model'] for r in s['ff2_validation']})==11
assert len({r['model'] for r in s['ff2_test']})==6
assert len({(r['model'],r['seed']) for r in s['ff2_test']})==10

class PaperTables(HTMLParser):
    def __init__(self):
        super().__init__();self.tables=[];self.depth=0;self.rows=None;self.cell=None
    def handle_starttag(self,t,a):
        if t=='table':
            if not self.depth:self.rows=[]
            self.depth+=1
        if self.depth==1:
            if t=='tr':self.row=[]
            if t in ('td','th'):self.cell=[]
    def handle_endtag(self,t):
        if self.depth==1:
            if t in ('td','th') and self.cell is not None:self.row.append(''.join(self.cell).strip());self.cell=None
            if t=='tr':self.rows.append(self.row)
        if t=='table':
            self.depth-=1
            if not self.depth:self.tables.append(self.rows)
    def handle_data(self,d):
        if self.depth==1 and self.cell is not None:self.cell.append(d)


p=argparse.ArgumentParser();p.add_argument('--paper-html',type=Path);a=p.parse_args()
result={'status':'PASS','best_observed_cells_independently_verified':checks,'numbered_tables':6,
        'published_rows_preserved':62,'benchmark_candidate_reports':68,'ff2_validation_models':11,
        'ff2_test_models':6,'ff2_test_checkpoints':10,'official_paper_numeric_check':'NOT_REQUESTED'}
if a.paper_html:
    html=a.paper_html.read_text();parser=PaperTables();parser.feed(html);count=0
    for num in [1,2,3]:
        expected=s['paper']['tables'][str(num)];width=4 if num==1 else 10
        matches=[]
        for t in parser.tables:
            rows=[]
            for r in t:
                if len(r)!=width+3:continue
                try:v=[float(x) for x in r[3:]]
                except ValueError:continue
                rows.append((r,v))
            if len(rows)==len(expected) and rows[0][1]==expected[0]['values_pct']:matches.append(rows)
        assert len(matches)==1,(num,len(matches))
        block=None
        for (row,values),exp in zip(matches[0],expected):
            block=row[0] or block
            assert block==exp['block'],(num,block,exp['block'])
            assert row[2]==exp['backbone']
            assert values==exp['values_pct'],(num,block,row,exp)
            normalize=lambda t:re.sub(r'\s+','',re.sub(r'\[\d+\]','',t.replace('’',"'").replace(' (published)',''))).lower()
            if row[1]=='Ours':assert exp['method']=='Ding et al. (published)'
            else:assert normalize(row[1])==normalize(exp['method']),(row[1],exp['method'])
            count+=width
    result.update(official_paper_numeric_check='PASS',official_numeric_values_checked=count,
                  bibliography_indices='Version-specific reference indices omitted; method identities checked.',
                  official_html_sha256=hashlib.sha256(a.paper_html.read_bytes()).hexdigest(),
                  official_url='https://arxiv.org/html/2511.10150v5')
(ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
