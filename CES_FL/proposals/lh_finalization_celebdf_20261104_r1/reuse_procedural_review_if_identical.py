#!/usr/bin/env python3
"""Rebind the prior L_H visual review only when every new preview sheet is byte-identical.

No reviewer decision or approval flag is created or edited. The source review's
AI reviewer and non-human limitations remain explicit in the output receipt.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

OLD_SOURCE = Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-lh-finalize-exec-352e837/CES_FL/src/ces_fl_head_ablation.py')
NEW_SOURCE = Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-ding10-prep-eb3eaa9/CES_FL/src/ces_fl_head_ablation_resnet50.py')
OLD_PREVIEW = Path('/home/sachin.chaudhary/ces_fl_runs/plain_head_s17_20261103T101418Z/preview_H')
OLD_REVIEW = Path('/home/sachin.chaudhary/ces_fl_runs/plain_head_s17_20261103T101418Z/review_H/review_PRIVATE.json')
OLD_QUALIFICATION = Path('/home/sachin.chaudhary/ces_fl_runs/plain_head_s17_20261103T101418Z/review_H/qualification.json')
EXPECTED_OLD_SOURCE_SHA = '08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177'
REVIEWED_FUNCTIONS = {
    'seed_for':'3335d608ac7a1e39e53d9cb08f667c7a2027d53426fd5c4123e2b93d8b8d1e1c',
    'load_pixels':'7961e927c925b45ddb733201a1e2f1f0235e0a90328afbd3eea0a0b2673ccdde',
    'intervention':'dff292f84d47b7be9a30a3a7152f5a083f168456525368b6e08ea01ebc89f62b',
    'group_inventory':'5b0bff3d4c65a7fda31a60b40bac995d9138f4be16084398e8ae6f8a5e107de1',
    'auxiliary':'300be23df93434f16dfe1e638d0a053d302a7a9b36e844997dfd2dcb7c7e300d',
    'Originals':'34f84761fe087bd0a3cdc5f73bcee29faff1a06e22b22a2f4080f96b013eaa00',
    'image_uint8':'12377dd1db07c4ff6ce84eab9632154a2ad11ac1dbd347dc47a16468be0e2a1b',
    'preview':'793e5fec7e4e93148a83f3614209d5c767eae3d76e7f6608b4d5799fd3fa01be',
    'qualify':'6419938d91185959846464b48ac1f6e5d21dceac2bf8d925833ac0333112fad3',
    'check_qualification':'db1bddde2add1b1cd9fc2b0cfb50772535ce86868bd0c46335d99a3689ec695b',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def func_hash(path,name):
    text=Path(path).read_text();tree=ast.parse(text)
    node=next((n for n in tree.body if getattr(n,'name',None)==name),None)
    if node is None:raise ValueError(f'Missing reviewed function {name}')
    return hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()


def load_module(path):
    spec=importlib.util.spec_from_file_location('cesfl_review_rebind_head',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-preview',required=True);p.add_argument('--derived-review',required=True);p.add_argument('--qualification',required=True)
    args=p.parse_args()
    if sha(OLD_SOURCE)!=EXPECTED_OLD_SOURCE_SHA:raise ValueError('Old reviewed source changed')
    oldq=json.loads(OLD_QUALIFICATION.read_text());oldr=json.loads(OLD_REVIEW.read_text())
    oldpv=OLD_PREVIEW/'preview_PRIVATE.json';oldp=json.loads(oldpv.read_text())
    newdir=Path(args.new_preview).resolve();newpv_path=newdir/'preview_PRIVATE.json';newp=json.loads(newpv_path.read_text())
    if oldq['status']!='OPERATOR_REVIEW_RECORDED' or oldq['source_sha256']!=EXPECTED_OLD_SOURCE_SHA:
        raise ValueError('Historical qualification is invalid')
    if oldq['config_digest']!=oldp['config_digest'] or Path(oldq['preview_file']).resolve()!=oldpv.resolve():
        raise ValueError('Historical qualification does not bind its preview/config')
    if oldq['preview_sha256']!=sha(oldpv) or oldq['review_sha256']!=sha(OLD_REVIEW):
        raise ValueError('Historical review receipts changed')
    if sha(OLD_SOURCE)!=EXPECTED_OLD_SOURCE_SHA:raise ValueError('Historical source receipt changed')
    function_checks={name:{'old_sha256':func_hash(OLD_SOURCE,name),'new_sha256':func_hash(NEW_SOURCE,name)}
                     for name in REVIEWED_FUNCTIONS}
    if any(v['old_sha256']!=REVIEWED_FUNCTIONS[k] or v['new_sha256']!=REVIEWED_FUNCTIONS[k]
           for k,v in function_checks.items()):raise ValueError('Procedural image/preview code differs from the reviewed recipe')
    oldcases={r['id']:r for r in oldp['cases']};newcases={r['id']:r for r in newp['cases']}
    if set(oldcases)!=set(newcases):raise ValueError('Review case coverage changed')
    for cid,old in oldcases.items():
        new=newcases[cid]
        for field in ('image','image_sha256','settings','mask_area_fraction','groups','sheet_sha256'):
            if old[field]!=new[field]:raise ValueError(f'New preview differs for {cid}/{field}')
        if sha(old['sheet'])!=old['sheet_sha256'] or sha(new['sheet'])!=new['sheet_sha256']:
            raise ValueError(f'Preview sheet bytes changed for {cid}')
        if sha(old['sheet'])!=sha(new['sheet']):raise ValueError(f'Visual-review sheet is not byte-identical for {cid}')
    old_review_cases=oldr.get('cases',[])
    if {r['id'] for r in old_review_cases}!=set(newcases):raise ValueError('Review decisions do not cover the new preview')
    # Copy recorded decisions and notes exactly; only attach the hash of the verified identical new preview.
    derived={'preview_sha256':sha(newpv_path),'cases':old_review_cases,
        'reuse_attestation':{'source_review':str(OLD_REVIEW),'source_qualification_sha256':sha(OLD_QUALIFICATION),
            'all_case_sheet_bytes_identical':True,'all_procedural_functions_identical':True,
            'review_decisions_modified':False,'human_review_claimed':False,
            'reviewer_limitation':'The recorded reviewer is an AI visual reviewer, explicitly not human approval.'}}
    out=Path(args.derived_review).resolve()
    if out.exists():raise ValueError('Preserve existing derived review')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(derived,indent=2)+'\n')
    if Path(args.qualification).exists():raise ValueError('Preserve existing qualification')
    head=load_module(NEW_SOURCE)
    head.qualify(newdir,out,args.qualification)
    report={'status':'REUSED_REVIEW_BOUND_TO_IDENTICAL_PREVIEW','case_count':len(newcases),
        'old_review_sha256':sha(OLD_REVIEW),'derived_review_sha256':sha(out),
        'new_preview_sha256':sha(newpv_path),'qualification_sha256':sha(args.qualification),
        'all_reviewed_sheets_byte_identical':True,'human_review_claimed':False,
        'real_forgery_validity_proven':False,'function_hashes':function_checks}
    Path(args.qualification).with_suffix('.reuse_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
