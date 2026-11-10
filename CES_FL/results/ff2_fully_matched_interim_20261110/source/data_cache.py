"""Parallel fresh byte/decode audit and canonical metadata cache; TRAIN/SELECT only."""
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time

from PIL import Image

ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_fully_matched_20261110_r1')
OLD=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_lh_ps_20261106_r1/preflight')
AXES=('gender','race','age_bin','skin_bin','gender_race','ids8')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def normalise(path,trusted):
    path=Path(path)
    with path.open(newline='') as f:rows=list(csv.DictReader(f))
    seen=set()
    for r in rows:
        image=r['image']
        # The old full canonical inventory establishes this literal path's
        # canonical identity. Resolve only a genuinely different spelling.
        if image not in trusted:image=str((path.parent/image).resolve()) if not Path(image).is_absolute() else str(Path(image).resolve())
        if image not in trusted:raise ValueError('Image absent from the prior frozen canonical inventory')
        if image in seen:raise ValueError('Duplicate image membership')
        seen.add(image);r['image']=image;r['label']=int(r['label'])
        if r['label'] not in (0,1):raise ValueError('Invalid authenticity label')
        for key in ('identity_ids','source_ids'):
            values=sorted({x.strip() for x in r[key].split(';') if x.strip()})
            if not values:raise ValueError('Empty contributor metadata')
            r[key]=';'.join(values)
        for key in AXES:r[key]=(r.get(key) or 'unknown').strip() or 'unknown'
        r['family']=r.get('family') or 'unknown'
        r['image_sha256']=trusted[image]
    return rows

def audit_item(item):
    split,row=item;data=Path(row['image']).read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    if digest!=row['image_sha256']:raise ValueError('Frozen image bytes changed: '+row['image'])
    with Image.open(io.BytesIO(data)) as im:
        im.convert('RGB').load()
        if min(im.size)<2:raise ValueError('Invalid decoded image')
    return {'split':split,'image':row['image'],'sha256':digest}

def main():
    old=json.loads((OLD/'PREFLIGHT.json').read_text())
    if old['status']!='PASS' or sha(OLD/'inventory_PRIVATE.json')!=old['inventory_sha256']:
        raise ValueError('Prior canonical inventory not verified')
    previous=json.loads((OLD/'inventory_PRIVATE.json').read_text())
    trusted={x['image']:x['sha256'] for x in previous}
    train=normalise('/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/train.csv',trusted)
    val=normalise('/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/val.csv',trusted)
    assert len(train)==218266 and len(val)==31107
    for key in ('image','identity_ids','source_ids'):
        a={v for r in train for v in r[key].split(';')};b={v for r in val for v in r[key].split(';')}
        if a&b:raise ValueError('Cross-split '+key+' overlap')
    out=ROOT/'shared_input_audit'
    out.mkdir(exist_ok=True)
    items=[('train',r) for r in train]+[('val',r) for r in val]
    started=time.monotonic();inventory=[]
    with ThreadPoolExecutor(max_workers=32) as pool:
        for i,result in enumerate(pool.map(audit_item,items),1):
            inventory.append(result)
            if i%10000==0:print(json.dumps({'decoded':i,'total':len(items),'seconds':time.monotonic()-started}),flush=True)
    hashes={split:{x['sha256'] for x in inventory if x['split']==split} for split in ['train','val']}
    if hashes['train']&hashes['val']:raise ValueError('Cross-split byte overlap')
    inventory_path=out/'inventory_PRIVATE.json'
    inventory_path.write_text(json.dumps(inventory,indent=2)+'\n')
    cache=out/'rows_PRIVATE.json';cache.write_text(json.dumps({'train':train,'val':val})+'\n')
    record={'status':'PASS','decoded_images':len(inventory),'train_rows':len(train),'val_rows':len(val),
        'inventory_sha256':sha(inventory_path),'row_cache_sha256':sha(cache),'row_cache_path':str(cache),
        'source_sha256':sha('/home/sachin.chaudhary/ces_fl_sources/ff2-fully-matched-20261110/src/core.py'),
        'cache_builder_sha256':sha(__file__),'fresh_full_decode':True,'fresh_all_image_hashes':True,
        'prior_canonical_inventory_sha256':old['inventory_sha256'],'cross_split_content_overlap':0,
        'test_read':False,'audit_threads':32,'elapsed_seconds':time.monotonic()-started}
    (out/'PREFLIGHT.json').write_text(json.dumps(record,indent=2)+'\n')
    print('FULL_PARALLEL_AUDIT_PASS',len(inventory),flush=True)

if __name__=='__main__':main()
