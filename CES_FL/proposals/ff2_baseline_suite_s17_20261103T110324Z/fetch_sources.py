#!/usr/bin/env python3
"""Explicit source-only fetch. Never downloads weights or runs upstream programs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent


def fetch(root, records):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in records:
        out = root / entry['id']
        out.mkdir(parents=True,exist_ok=True)
        metadata=out/'SOURCE_PIN.json'
        if metadata.exists() and json.loads(metadata.read_text())!={'repository':entry['repository'],'commit':entry['commit']}:
            raise ValueError('Existing source tree is not pinned: '+str(out))
        selected=[]
        for relative,info in entry['files'].items():
            if relative.endswith(('.pth','.pt','.pyc')) or '__pycache__' in Path(relative).parts:
                continue
            if Path(relative).is_absolute() or '..' in Path(relative).parts:raise ValueError('Unsafe source path')
            selected.append((relative,info))
        def fetch_one(item):
            relative,info=item
            p = out / relative
            if p.exists():data=p.read_bytes()
            else:
                repo=entry['repository'].removeprefix('https://github.com/').removesuffix('.git')
                url='https://raw.githubusercontent.com/'+repo+'/'+entry['commit']+'/'+urllib.parse.quote(relative)
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(url,timeout=45) as response:data=response.read()
                        break
                    except Exception:
                        if attempt==2:raise
                        time.sleep(attempt+1)
            blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if blob != info['git_blob']: raise ValueError('Changed upstream source: '+str(p))
            if not p.exists():
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        # Fetch specific blobs: a filtered Git clone can still fetch excluded
        # binary history, so do not clone repositories that bundle checkpoints.
        with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(fetch_one,selected))
        metadata.write_text(json.dumps({'repository':entry['repository'],'commit':entry['commit']},indent=2)+'\n')
        results.append({'id':entry['id'],'commit':entry['commit'],'path':str(out),'verified_files':len(selected)})
    receipt = root / 'SOURCE_FETCH_VERIFIED.json'
    receipt.write_text(json.dumps({'status':'PASS','sources':results,'weights_downloaded':False},indent=2)+'\n')
    return results


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    a=p.parse_args()
    print(json.dumps(fetch(a.root,json.loads((HERE/'upstreams.lock.json').read_text())['repositories']),indent=2))
