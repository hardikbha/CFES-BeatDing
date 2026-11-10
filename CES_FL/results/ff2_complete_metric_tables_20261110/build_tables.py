"""Build complete tables from existing public aggregate receipts; no inference."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from xml.sax.saxutils import escape, quoteattr
import zipfile

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'ff2_fully_matched_final_20261110'
METHODS=('bce','dag','daw','ding_sfd_gda','lh','lh_ramp','lh_low')
SEEDS=(7,17,27)
AXES=('gender','race','gender_race','age_bin','skin_bin','ids8')
NAMES={'bce':'BCE','dag':'DAG-FDD','daw':'DAW-FDD','ding_sfd_gda':'Ding adaptation',
       'lh':'Our full L_H','lh_ramp':'Our ramp','lh_low':'Our low response'}
AXIS_NAMES={'gender':'Gender','race':'Race','gender_race':'Gender x race','age_bin':'Age',
            'skin_bin':'Skin tone','ids8':'IDS8'}
OVERALL=('AUC_pct','achieved_FPR_pct','FNR_pct')
METRICS=('F_FPR_100','F_DP_100','es_AUC_pct','FPR_gap_pp','FNR_gap_pp','AUC_gap_pp',
         'F_MEO_max_pp','F_OAE_pp','worst_group_AUC_pct')
HIGH={'AUC_pct','es_AUC_pct','worst_group_AUC_pct'}
LABELS={'AUC_pct':'AUC %','achieved_FPR_pct':'FPR %','FNR_pct':'FNR %','F_FPR_100':'100F_FPR',
        'F_DP_100':'100F_DP','es_AUC_pct':'es-AUC %','FPR_gap_pp':'FPR gap pp','FNR_gap_pp':'FNR gap pp',
        'AUC_gap_pp':'AUC gap pp','F_MEO_max_pp':'MEO-max gap pp','F_OAE_pp':'Accuracy gap pp','worst_group_AUC_pct':'Worst-group AUC %'}


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def csv_rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def write_csv(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9),(a,b)
def sheet(name,headers,rows,style=None):return dict(name=name,headers=headers,rows=rows,style=style)
def letters(index):
    out=''
    while index:index,remainder=divmod(index-1,26);out=chr(65+remainder)+out
    return out


def xlsx(path,sheets):
    """Small standards-based OOXML workbook: no third-party dependency."""
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    styles=f'''<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="{ns}">
      <numFmts count="1"><numFmt numFmtId="164" formatCode="0.0000"/></numFmts>
      <fonts count="3"><font><sz val="11"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>
      <fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF17365D"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFE2F0D9"/></patternFill></fill></fills>
      <borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs>
      <cellXfs count="5"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0"/><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="164" fontId="2" fillId="3" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf></cellXfs>
      <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        overrides=''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(sheets)+1))
        z.writestr('[Content_Types].xml','<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'+overrides+'</Types>')
        z.writestr('_rels/.rels','<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml',f'<?xml version="1.0"?><workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+''.join(f'<sheet name={quoteattr(s["name"])} sheetId="{i}" r:id="rId{i}"/>' for i,s in enumerate(sheets,1))+'</sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(sheets)+1))+f'<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        z.writestr('xl/styles.xml',styles)
        for i,s in enumerate(sheets,1):
            data=[s['headers'],*s['rows']];xmlrows=[];ncols=len(s['headers'])
            for ri,row in enumerate(data,1):
                cells=[]
                for ci,value in enumerate(row,1):
                    style=1 if ri==1 else 2 if isinstance(value,float) else 0
                    if ri>1 and s.get('style') is not None:style=s['style'](ri-2,ci-1,style)
                    if s['name']=='Read me' and ri>1:style=4
                    ref=f'{letters(ci)}{ri}'
                    if isinstance(value,(int,float)) and not isinstance(value,bool):
                        assert math.isfinite(value)
                        cells.append(f'<c r="{ref}" s="{style}"><v>{value}</v></c>')
                    else:cells.append(f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(str(value))}</t></is></c>')
                xmlrows.append(f'<row r="{ri}"'+(' ht="48" customHeight="1"' if s['name']=='Read me' and ri>1 else '')+'>'+''.join(cells)+'</row>')
            cols=f'<col min="1" max="1" width="24" customWidth="1"/><col min="2" max="{ncols}" width="18" customWidth="1"/>'
            if s['name']=='Read me':cols='<col min="1" max="1" width="28" customWidth="1"/><col min="2" max="2" width="110" customWidth="1"/>'
            end=f'{letters(ncols)}{len(data)}'
            z.writestr(f'xl/worksheets/sheet{i}.xml',f'<?xml version="1.0"?><worksheet xmlns="{ns}"><dimension ref="A1:{end}"/><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols>{cols}</cols><sheetData>'+''.join(xmlrows)+f'</sheetData><autoFilter ref="A1:{end}"/></worksheet>')


def main():
    original=csv_rows(SOURCE/'TEST_PER_SEED_ALL_AXES.csv');lookup={};groups=[];native_hashes={}
    for seed in SEEDS:
        for method in METHODS:
            path=SOURCE/'native'/f'{method}_s{seed}'/'TEST_METRICS.json';native_hashes[str(path.relative_to(ROOT.parent))]=sha(path)
            native=read(path)['calibrated']
            for axis in AXES:
                old=next(r for r in original if r['method']==method and int(r['seed'])==seed and r['axis']==axis)
                r=dict(method=method,seed=seed,axis=axis,**{k:float(v) for k,v in old.items() if k not in ('method','seed','axis')})
                g=native['axes'][axis]
                for dest,src in [('FPR_gap_pp','fpr_gap'),('FNR_gap_pp','fnr_gap'),('AUC_gap_pp','auc_gap')]:r[dest]=100*g[src]
                lookup[(method,seed,axis)]=r
                for name,v in g['groups'].items():
                    groups.append(dict(method=method,seed=seed,axis=axis,group=name,eligible=name in g['eligible_groups'],
                        n=v['n'],n_real=v['n_real'],n_fake=v['n_fake'],AUC_pct=100*v['auc'],FPR_pct=100*v['fpr'],FNR_pct=100*v['fnr'],
                        positive_rate_pct=100*v['positive_rate'],accuracy_pct=100*(v['n_real']*(1-v['fpr'])+v['n_fake']*(1-v['fnr']))/v['n']))
    long=[lookup[(m,s,a)] for s in SEEDS for m in METHODS for a in AXES]
    write_csv('ALL_METRICS_LONG.csv',long);write_csv('GROUP_RATES_AND_SUPPORT.csv',groups)
    overall=[dict(method=m,seed=s,**{k:lookup[(m,s,'gender_race')][k] for k in OVERALL}) for s in SEEDS for m in METHODS]
    wide=[dict(method=m,seed=s,**{a:lookup[(m,s,a)]['F_FPR_100'] for a in AXES}) for s in SEEDS for m in METHODS]
    write_csv('OVERALL_ALL_SEEDS.csv',overall);write_csv('FPR_DISPARITY_ALL_SEEDS.csv',wide)
    leaders=[];leader_lookup={};wins={(m,s):[] for m in METHODS for s in SEEDS}
    cells=[('overall',k) for k in OVERALL]+[(a,k) for a in AXES for k in METRICS]
    for seed in SEEDS:
        for axis,key in cells:
            vals={m:lookup[(m,seed,'gender_race' if axis=='overall' else axis)][key] for m in METHODS}
            best=(max if key in HIGH else min)(vals.values());winning=[m for m,v in vals.items() if math.isclose(v,best,rel_tol=0,abs_tol=1e-10)]
            leader_lookup[(seed,axis,key)]=set(winning)
            leaders.append(dict(seed=seed,axis=axis,metric=key,direction='higher' if key in HIGH else 'lower',best_value=best,methods=';'.join(winning)))
            for method in winning:wins[(method,seed)].append(axis+'/'+key)
    write_csv('BEST_PER_METRIC.csv',leaders)
    summary=[dict(method=m,seed=s,leading_or_tied_cells=len(wins[(m,s)]),cells_checked=len(cells),leads_every_metric=len(wins[(m,s)])==len(cells)) for s in SEEDS for m in METHODS]
    assert not any(r['leads_every_metric'] for r in summary);write_csv('LEADER_SUMMARY.csv',summary)
    means=[]
    for method in METHODS:
        for axis,key in cells:
            vals=[lookup[(method,s,'gender_race' if axis=='overall' else axis)][key] for s in SEEDS]
            means.append(dict(method=method,axis=axis,metric=key,n_seeds=3,mean=statistics.mean(vals),sample_SD=statistics.stdev(vals)))
    write_csv('THREE_SEED_MEANS_ALL_METRICS.csv',means)
    families=csv_rows(SOURCE/'TEST_PER_FAMILY.csv');write_csv('FAMILY_RESULTS.csv',families)
    def fmt(value,best=False):
        text=f'{value:.4f}'
        return '**'+text+'**' if best else text
    def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']+['---:']*(len(headers)-1))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
    def values_for(seed,axis,keys):
        return [[NAMES[m]]+[fmt(lookup[(m,seed,'gender_race' if axis=='overall' else axis)][k],m in leader_lookup[(seed,axis,k)]) for k in keys] for m in METHODS]
    md=['# Complete FF2 TEST tables','', 'All seven methods, all three seeds. Bold marks exact best/tied values within the same seed/axis/metric. These are descriptive point estimates, not significance claims.','', 'AUC, FPR, FNR and es-AUC are percentages; gaps are percentage points. 100F_FPR is a sum of absolute deviations from pooled FPR; 100F_DP is a prediction-rate gap. MEO-max is the larger FPR/FNR gap; accuracy gap is max–min group accuracy.','']
    for seed in SEEDS:
        md+=['## Seed '+str(seed),'','### Overall','',table(['Method']+[LABELS[k]+(' ↑' if k in HIGH else ' ↓') for k in OVERALL],values_for(seed,'overall',OVERALL))]
        for axis in AXES:
            md+=['### '+AXIS_NAMES[axis],'',table(['Method']+[LABELS[k]+(' ↑' if k in HIGH else ' ↓') for k in METRICS],values_for(seed,axis,METRICS))]
    (ROOT/'ALL_METRICS.md').write_text('\n'.join(md))
    notes=[['Scope','FF2 natural-population TEST, 87,564 images. Seven controlled adaptations, seeds7/17/27, ten downstream epochs each.'],
        ['Uncertainty','Three downstream seeds share one P50 encoder. Sample SD is conditional downstream variation, not a confidence interval.'],
        ['Axes','Gender, race, gender×race, estimated age, estimated skin tone, IDS8. Skin tone is the color-related axis; there is no independent color axis.'],
        ['IDS8','Eight supplied FairFace-derived gender×coarse age×coarse race cells; distinct from the eight native gender×race groups.'],
        ['FOV wording','No saved metric is named FOV. FPR disparity, demographic parity, equity-scaled AUC, equalized-odds gaps and accuracy gaps are included.'],
        ['Age / skin labels','Automatic apparent-attribute estimates; missing/unknown bins are excluded from eligible group summaries. All group rates and denominators remain available.'],
        ['Metrics','All rates/AUC are displayed ×100. 100F_FPR sums absolute group deviations from pooled FPR. Do not compare its scale across axes with different group counts.'],
        ['Thresholds','One global SELECT-real10%FPR threshold per checkpoint, frozen for TEST. Achieved TEST FPR differs; no TEST threshold retuning.'],
        ['Best values','Green/bold cells compare all seven methods at a fixed seed and axis, using unrounded numbers with absolute tie tolerance1e-10. Counts are descriptive, correlated, and not superiority tests.'],
        ['Ramp / low','All six selected epoch2 before response loss activation; metrics are identical within seed. Do not attribute those results to response learning.'],
        ['Protocol','These are controlled adaptations, including Ding SFD+GDA in the shared representation, not each native paper recipe.'],
        ['Completeness','126 method/seed/axis rows with12numeric metrics each; 21 overall rows; 651 group rows; 147 family rows; 399 distinct mean/SD rows; 171 metric-leader rows.'],
        ['Claim','No single seed/method leads every reported metric. Full L_H seed17 leads age/skin FPR disparity but has lower AUC and higher FNR. Retain all seeds.'],
        ['Source','Derived solely from existing published native aggregate reports; no model execution, threshold fitting, new TEST inference or private prediction access.']]
    sheets=[sheet('Read me',['Topic','Explanation'],notes)]
    def table_style(data,axis_for,keys,first_numeric_col):
        def style(ri,ci,default):
            if ci<first_numeric_col:return default
            method,seed,axis=data[ri];key=keys[ci-first_numeric_col]
            return 3 if method in leader_lookup[(seed,'overall' if key in OVERALL else axis_for(axis),key)] else default
        return style
    overall_data=[(m,s,'overall') for s in SEEDS for m in METHODS]
    sheets.append(sheet('Overall',['Method','Seed']+[LABELS[k] for k in OVERALL],[[NAMES[r['method']],r['seed']]+[r[k] for k in OVERALL] for r in overall],table_style(overall_data,lambda x:x,OVERALL,2)))
    sheets.append(sheet('FPR disparity',['Method','Seed']+[AXIS_NAMES[a] for a in AXES],[[NAMES[r['method']],r['seed']]+[r[a] for a in AXES] for r in wide],lambda ri,ci,d:3 if ci>=2 and wide[ri]['method'] in leader_lookup[(wide[ri]['seed'],AXES[ci-2],'F_FPR_100')] else d))
    for seed in SEEDS:
        data=[(m,seed,a) for a in AXES for m in METHODS];keys=OVERALL+METRICS
        sheets.append(sheet('All metrics seed '+str(seed),['Method','Seed','Axis']+[LABELS[k] for k in keys],[[NAMES[m],seed,AXIS_NAMES[a]]+[lookup[(m,seed,a)][k] for k in keys] for m,seed,a in data],table_style(data,lambda x:x,keys,3)))
    for name,data in [('Means and SD',means),('Best per metric',leaders),('Leader summary',summary),('Group support and rates',groups),('Family results',families)]:
        sheets.append(sheet(name,list(data[0]),[list(r.values()) for r in data]))
    xlsx(ROOT/'FF2_COMPLETE_RESULTS.xlsx',sheets)
    README=['# FF2 complete demographic and detection results','', '**No seed tops every metric.** This packet expands the completed21-run comparison across every saved axis. It preserves all methods and seeds rather than selecting favorable cells.','',
        '- [Download Excel workbook](FF2_COMPLETE_RESULTS.xlsx): filtered/frozen tables, all seeds, all axes, means/SDs, metric leaders, group support and family results. Green cells mark best/tied values within each seed/axis/metric.',
        '- [Read every metric table](ALL_METRICS.md), [full numeric CSV](ALL_METRICS_LONG.csv), [three-seed means/SDs](THREE_SEED_MEANS_ALL_METRICS.csv), [group rates/support](GROUP_RATES_AND_SUPPORT.csv).',
        '- [Metric leaders](BEST_PER_METRIC.csv), [leader summary](LEADER_SUMMARY.csv), [family results](FAMILY_RESULTS.csv). Leader counts include correlated metrics and ties; they are not a scientific win count or statistical test.','',
        'Age and skin-tone metrics exist. Skin tone is the color-related axis; no separate color metric was computed. IDS8 represents the separate eight supplied gender×age×race cells. There is no native metric named FOV; accuracy-gap fairness (F_OAE) is included.','',
        'All21runs share the training/evaluation recipe, with declared objective/SFD differences. Ding is the controlled adaptation. Each threshold was frozen from SELECT-real calibration. All ramp/low selected epoch2, before response losses activate. Three-seed SD is conditional on one shared pretrained encoder; no significance claim or independent-pretraining replication. Age/skin are automatic estimates.','',
        'The tables below show only100F_FPR by axis (lower is better). Bold uses full precision; visually equal rounded numbers need not be ties. The workbook and full Markdown include F_DP, es-AUC, FPR/FNR/AUC gaps, MEO-max, accuracy gap, worst-group AUC and overall AUC/FPR/FNR.','']
    for seed in SEEDS:
        rr=[]
        for m in METHODS:rr.append([NAMES[m]]+[fmt(lookup[(m,seed,a)]['F_FPR_100'],m in leader_lookup[(seed,a,'F_FPR_100')]) for a in AXES])
        README+=['## Seed '+str(seed),'',table(['Method']+[AXIS_NAMES[a] for a in AXES],rr)]
    README+=['','Seed17 full L_H has the lowest age and skin FPR disparities, with lower AUC/higher miss rate. Seed27 ramp/low has the lowest race, joint, skin and IDS8 FPR disparities; its skin advantage over DAW is only0.00451points. No result supports topping every metric.','',
        '[Original final-results verification](../ff2_fully_matched_final_20261110/VERIFICATION.json), [source provenance](PROVENANCE.json), [builder](build_tables.py), [artifact hashes](SHA256.json). This packet computes tables from stored aggregates, not new predictions.']
    text='\n'.join(README).replace('completed21-run','completed 21-run').replace('All21runs','All 21 runs').replace('epoch2','epoch 2').replace('only100F_FPR','only 100F_FPR').replace('Seed17','Seed 17').replace('Seed27','Seed 27').replace('only0.00451points','only 0.00451 points')
    (ROOT/'README.md').write_text(text)
    provenance=dict(observed_management_utc=datetime.now(timezone.utc).isoformat(),source_snapshot=str(SOURCE),source_native_aggregate_hashes=native_hashes,
        native_runs=21,method_seed_axis_rows=len(long),numeric_metrics_per_axis=len(OVERALL+METRICS),distinct_cells_per_seed=len(cells),leader_rows=len(leaders),mean_SD_rows=len(means),
        group_rows=len(groups),family_rows=len(families),workbook_sheets=[s['name'] for s in sheets],all_metric_leader_exists=False,new_inference=False,private_predictions_read=False)
    (ROOT/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({k:v for k,v in provenance.items() if k!='source_native_aggregate_hashes'},indent=2))


if __name__=='__main__':main()
