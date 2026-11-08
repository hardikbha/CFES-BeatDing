#!/usr/bin/env python3
"""Bind the already decoded, SHA-verified FF2 TRAIN/VAL inventory to a seed-only config."""
import argparse, hashlib, importlib.util, json, shutil
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--source',required=True)
p.add_argument('--config',required=True)
p.add_argument('--reference-config',required=True)
p.add_argument('--reference-preflight',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
src=Path(a.source).resolve(); config=json.loads(Path(a.config).read_text()); ref=json.loads(Path(a.reference_config).read_text())
pre=Path(a.reference_preflight); rec=json.loads((pre/'PREFLIGHT.json').read_text()); invp=pre/'inventory_PRIVATE.json'
modpath=src/'CES_FL/src/ces_fl_realfpr.py'
spec=importlib.util.spec_from_file_location('realfpr',modpath)
rf=importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
 return h.hexdigest()
if rec.get('status')!='PASS' or not rec.get('preflight_reused',False): raise SystemExit('reference TRAIN/VAL preflight not PASS/reused')
if rec.get('test_read') is not False: raise SystemExit('reference preflight touched TEST')
if rec.get('source_sha256')!=rf.ham.sha(rf.ham.__file__): raise SystemExit('reference source hash mismatch')
if sha(invp)!=rec.get('inventory_sha256'): raise SystemExit('reference inventory hash mismatch')
rf.check(config)
if rf.ham.sha(rf.ham.__file__)!=rec['source_sha256']: raise SystemExit('model source differs from reference preflight')
# Require exactly seed-only change. This keeps all scientific settings, data and operations fixed.
for k in set(config)|set(ref):
 if k=='seed': continue
 if config.get(k)!=ref.get(k): raise SystemExit(f'non-seed config difference: {k}')
if int(config['seed']) not in (7,27) or int(ref['seed'])!=17: raise SystemExit('unexpected seed pair')
for split,key in [('train','train_sha256'),('val','val_sha256')]:
 if sha(config[split])!=config[key] or config[split]!=ref[split] or config[key]!=ref[key]: raise SystemExit(f'{split} manifest mismatch')
# Recheck every TRAIN/VAL source byte hash against the already decoded baseline inventory.
inventory=json.loads(invp.read_text())
if len(inventory)!=rec.get('decoded_images'): raise SystemExit('inventory row count mismatch')
for i,row in enumerate(inventory):
 if row['split'] not in ('train','val') or sha(row['image'])!=row['sha256']:
  raise SystemExit(f'image inventory byte mismatch at row {i}')
out=Path(a.out)
if out.exists(): raise SystemExit(f'refusing existing preflight output: {out}')
out.mkdir(parents=True)
shutil.copy2(invp,out/'inventory_PRIVATE.json')
new={'status':'PASS','source_sha256':rec['source_sha256'],'config_digest':rf.rf_digest(config),
 'inventory_sha256':sha(out/'inventory_PRIVATE.json'),'decoded_images':rec['decoded_images'],
 'train_rows':rec['train_rows'],'val_rows':rec['val_rows'],'cross_split_content_overlap':rec['cross_split_content_overlap'],
 'test_read':False,'synthetic_fixture':False,'preflight_reused':True,
 'reuse_basis':'Seed-only config change; original full TRAIN/VAL decode receipt and inventory retained; each current TRAIN/VAL image byte SHA rechecked before rebinding.',
 'reference_preflight':str(pre/'PREFLIGHT.json'),'reference_inventory_sha256':rec['inventory_sha256'],
 'seed':int(config['seed']),'config_sha256':sha(a.config),'train_manifest_sha256':config['train_sha256'],'val_manifest_sha256':config['val_sha256']}
(out/'PREFLIGHT.json').write_text(json.dumps(new,indent=2)+'\n')
print(json.dumps(new,indent=2))
