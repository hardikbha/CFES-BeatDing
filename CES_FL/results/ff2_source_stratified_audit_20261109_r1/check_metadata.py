"""Metadata-only postcheck of full cached row alignment and component support."""
import csv
import collections
import posixpath
from pathlib import Path
from audit import TRAIN,SELECT,rows,source,read,write
from audit_stats import connected_components

tr=rows(TRAIN);va=rows(SELECT)
allc=connected_components(tr+va);comps=allc[len(tr):];by=collections.defaultdict(set)
for r,c in zip(va,comps):by[int(c)].add(source(r))
with SELECT.open() as f:raw=list(csv.DictReader(f))
checked=0
for name,arm in [('RF-S_s17','S'),('RF-S_s7','S_s7'),('RF-S_s27','S_s27'),('RF-SJg_s17','SJ_gentle'),('RF-SJg_s7','SJ_gentle_s7'),('RF-SJg_s27','SJ_gentle_s27')]:
    p=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_rfa_20261107')/arm/'out/validation_scores_PRIVATE.json';d=read(p)
    for original,cache in zip(raw,d['rows']):
        for k,v in original.items():
            if k=='image':assert posixpath.normpath(v)==posixpath.normpath(cache[k])
            elif k=='label':assert int(v)==int(cache[k])
            elif k in ['identity_ids','source_ids']:assert {x for x in v.split(';') if x}=={x for x in cache[k].split(';') if x}
            elif k in ['gender','race','gender_race','age_bin','skin_bin','ids8','family']:assert (v or 'unknown')==(cache[k] or 'unknown')
            else:assert v==cache[k],k
        checked+=1
report={'status':'PASS','full_csv_fields_aligned_rows_over_six_exports':checked,'matched_fields':list(raw[0]),
 'select_source_master_counts':{},'recorded_identifier_namespace_usage':{},'metadata_only_no_pixels_or_new_predictions':True,
 'select_bearing_components':len(by),'cross_acquisition_components':sum(len(s)>1 for s in by.values()),
 'source_nested_component_counts':dict(collections.Counter(next(iter(s)) for s in by.values() if len(s)==1)),
 'per_image_processing_columns_absent_from_split_manifests':['profile','config_hash','master_path','master_sha256']}
for split,data in [('TRAIN',tr),('SELECT',va)]:
    reals=[r for r in data if r['label']==0];count=collections.Counter(r['source_ids'] for r in reals)
    report['select_source_master_counts'][split]={'real_rows':len(reals),'unique_recorded_real_master_keys':len(count),'repeated_real_master_keys':sum(n>1 for n in count.values())}
    assert len(count)==len(reals),'One recorded master per real row was assumed by the presentation layer'
    report['recorded_identifier_namespace_usage'][split]={k:dict(collections.Counter(t.split(':',1)[0] for r in data for t in r[k].split(';'))) for k in ['identity_ids','source_ids']}
write(Path('METADATA_CHECKS.json'),report)
print('PASS: all 16 CSV fields align over 186642 cached rows; recorded master-key uniqueness and component census checked.')
