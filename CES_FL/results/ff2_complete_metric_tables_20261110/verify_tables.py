"""Read back every numeric worksheet cell and independently check metric leaders."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parent
METHODS=('bce','dag','daw','ding_sfd_gda','lh','lh_ramp','lh_low')
SEEDS=(7,17,27)
AXES=('gender','race','gender_race','age_bin','skin_bin','ids8')
OVERALL=('AUC_pct','achieved_FPR_pct','FNR_pct')
PER_AXIS=('F_FPR_100','F_DP_100','es_AUC_pct','FPR_gap_pp','FNR_gap_pp','AUC_gap_pp','F_MEO_max_pp','F_OAE_pp','worst_group_AUC_pct')
HIGH={'AUC_pct','es_AUC_pct','worst_group_AUC_pct'}
NAMES={'BCE':'bce','DAG-FDD':'dag','DAW-FDD':'daw','Ding adaptation':'ding_sfd_gda','Our full L_H':'lh','Our ramp':'lh_ramp','Our low response':'lh_low'}
AXIS_NAMES={'Gender':'gender','Race':'race','Gender x race':'gender_race','Age':'age_bin','Skin tone':'skin_bin','IDS8':'ids8'}


def rows(name):
    with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f))
def read(name):return json.loads((ROOT/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)


def main():
    long=rows('ALL_METRICS_LONG.csv');assert len(long)==126
    lookup={(r['method'],int(r['seed']),r['axis']):r for r in long};assert len(lookup)==126
    assert set(lookup)=={(m,s,a) for m in METHODS for s in SEEDS for a in AXES}
    leaders=rows('BEST_PER_METRIC.csv');assert len(leaders)==171
    winner_sets={};counts={(m,s):0 for m in METHODS for s in SEEDS}
    for r in leaders:
        seed=int(r['seed']);axis=r['axis'];metric=r['metric'];assert metric in OVERALL+PER_AXIS
        values={m:float(lookup[(m,seed,'gender_race' if axis=='overall' else axis)][metric]) for m in METHODS}
        best=(max if metric in HIGH else min)(values.values());close(best,r['best_value'])
        expected={m for m,v in values.items() if abs(v-best)<=1e-10};assert expected==set(r['methods'].split(';'))
        winner_sets[(seed,axis,metric)]=expected
        for m in expected:counts[(m,seed)]+=1
    for r in rows('LEADER_SUMMARY.csv'):
        count=counts[(r['method'],int(r['seed']))];assert int(r['leading_or_tied_cells'])==count
        assert int(r['cells_checked'])==57 and r['leads_every_metric']=='False' and count<57
    means=rows('THREE_SEED_MEANS_ALL_METRICS.csv');assert len(means)==399
    for r in means:
        vals=[float(lookup[(r['method'],s,'gender_race' if r['axis']=='overall' else r['axis'])][r['metric']]) for s in SEEDS]
        close(statistics.mean(vals),r['mean']);close(statistics.stdev(vals),r['sample_SD'])
    p=read('PROVENANCE.json');native_checked=0
    for relative,h in p['source_native_aggregate_hashes'].items():
        f=ROOT.parent/relative
        if not f.exists():continue
        assert sha(f)==h
        parts=f.parent.name.rsplit('_s',1);method,seed=parts[0],int(parts[1]);report=json.loads(f.read_text())['calibrated']
        for axis in AXES:
            row=lookup[(method,seed,axis)];g=report['axes'][axis]
            for metric,key in [('FPR_gap_pp','fpr_gap'),('FNR_gap_pp','fnr_gap'),('AUC_gap_pp','auc_gap')]:close(row[metric],100*g[key])
        native_checked+=1
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    workbook=ROOT/'FF2_COMPLETE_RESULTS.xlsx';numeric_checked=0;all_best_marking=0
    with zipfile.ZipFile(workbook) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith(('.xml','.rels')):ET.fromstring(z.read(name))
        wb=ET.fromstring(z.read('xl/workbook.xml'));names=[e.attrib['name'] for e in wb.findall('s:sheets/s:sheet',ns)]
        assert names==p['workbook_sheets'] and len(names)==11
        def sheet(name):
            index=names.index(name)+1;xml=ET.fromstring(z.read(f'xl/worksheets/sheet{index}.xml'));out=[];styles=[]
            assert xml.find('s:sheetViews/s:sheetView/s:pane',ns).attrib['state']=='frozen'
            for row in xml.findall('s:sheetData/s:row',ns):
                cells=[];sty=[]
                for c in row.findall('s:c',ns):
                    cells.append(''.join(t.text or '' for t in c.findall('s:is/s:t',ns)) if c.attrib.get('t')=='inlineStr' else float(c.find('s:v',ns).text))
                    sty.append(int(c.attrib.get('s',0)))
                out.append(cells);styles.append(sty)
            return out,styles
        for seed in SEEDS:
            data,styles=sheet('All metrics seed '+str(seed));assert len(data)==43
            for i,r in enumerate(data[1:],1):
                method=NAMES[r[0]];axis=AXIS_NAMES[r[2]];assert int(r[1])==seed
                src=lookup[(method,seed,axis)]
                for col,metric in enumerate(OVERALL+PER_AXIS,3):
                    close(r[col],src[metric]);numeric_checked+=1
                    is_best=method in winner_sets[(seed,'overall' if metric in OVERALL else axis,metric)]
                    assert (styles[i][col]==3)==is_best;all_best_marking+=1
        for name,csvname in [('Means and SD','THREE_SEED_MEANS_ALL_METRICS.csv'),('Best per metric','BEST_PER_METRIC.csv'),('Leader summary','LEADER_SUMMARY.csv'),('Group support and rates','GROUP_RATES_AND_SUPPORT.csv'),('Family results','FAMILY_RESULTS.csv')]:
            data,_=sheet(name);expected=rows(csvname);assert len(data)==len(expected)+1
            assert data[0]==list(expected[0])
            for actual,row in zip(data[1:],expected):
                for value,raw in zip(actual,row.values()):
                    if isinstance(value,float):close(value,raw);numeric_checked+=1
                    else:assert value==str(raw)
    assert len(rows('GROUP_RATES_AND_SUPPORT.csv'))==651 and len(rows('FAMILY_RESULTS.csv'))==147
    if (ROOT/'SHA256.json').exists():
        for n,h in read('SHA256.json').items():assert sha(ROOT/n)==h,n
    print(json.dumps(dict(status='PASS',method_seed_axis_rows=126,distinct_metric_cells_per_seed=57,
        verified_metric_leader_rows=171,mean_SD_rows=399,native_aggregate_hashes_checked=native_checked,
        workbook_sheets=11,numeric_workbook_cells_checked=numeric_checked,best_cell_styles_checked=all_best_marking,
        group_rows=651,family_rows=147,method_leading_every_metric=False,new_inference=False,private_predictions_read=False),indent=2))


if __name__=='__main__':main()
