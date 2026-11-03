#!/usr/bin/env python3
"""Audit optional UCF/PG Specific targets on TRAIN only; never infer them."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',type=Path,required=True);p.add_argument('--specific-csv',type=Path)
    p.add_argument('--provenance',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Preserve previous audit')
    expected=json.loads((HERE/'protocol.json').read_text())['splits']['train']
    if hashlib.sha256(a.train.read_bytes()).hexdigest()!=expected['sha256']:raise ValueError('Wrong TRAIN manifest')
    rows=list(csv.DictReader(a.train.open()));paths={r['image']:r for r in rows}
    if len(rows)!=expected['rows'] or len(paths)!=len(rows):raise ValueError('Membership changed')
    result={'train_sha256':expected['sha256'],'train_rows':len(rows),'test_read':False,
            'status':'BLOCKED_SPECIFIC_LABEL_PROVENANCE','training_authorized_by_this_audit':False}
    if a.specific_csv:
        if not a.provenance:raise ValueError('Document canonical target semantics and provenance')
        provenance=json.loads(a.provenance.read_text())
        digest=hashlib.sha256(a.specific_csv.read_bytes()).hexdigest()
        if provenance.get('specific_csv_sha256')!=digest or not provenance.get('source_reference') or not provenance.get('label_semantics'):
            raise ValueError('Specific-label provenance not bound')
        labels=list(csv.DictReader(a.specific_csv.open()));lookup={r['image']:int(r['specific_label']) for r in labels}
        if len(lookup)!=len(labels) or set(lookup)!=set(paths):raise ValueError('Specific labels must cover all TRAIN exactly')
        if any((int(paths[k]['label'])==0 and v!=0) or (int(paths[k]['label'])==1 and v<=0) for k,v in lookup.items()):
            raise ValueError('Real must be0; fake Specific labels must be positive canonical categories')
        codes=sorted(set(lookup.values()))
        if codes!=list(range(len(codes))):raise ValueError('Use a documented contiguous class mapping')
        result.update(status='METADATA_STRUCTURALLY_VERIFIED_NATIVE_ADAPTER_STILL_REQUIRED',specific_sha256=digest,
                      provenance_sha256=hashlib.sha256(a.provenance.read_bytes()).hexdigest(),
                      specific_counts=dict(Counter(lookup.values())),native_four_way_head_compatible=len(codes)==4,
                      independently_verified_label_truth=False)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
