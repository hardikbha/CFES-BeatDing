#!/usr/bin/env python3
"""Clone pinned source-only sparse worktrees; never run upstream programs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import urllib.parse

HERE = Path(__file__).resolve().parent


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs).stdout.strip()


def clone(root, entries):
    root.mkdir(parents=True, exist_ok=True)
    receipts = []
    for entry in entries:
        dst = root / entry['id']
        if not dst.exists():
            run(['git', 'init', '-q', str(dst)])
            run(['git', '-C', str(dst), 'remote', 'add', 'origin', entry['repository']])
        if run(['git', '-C', str(dst), 'remote', 'get-url', 'origin']) != entry['repository']:
            raise ValueError('Existing repository origin differs')
        run(['git', '-C', str(dst), 'config', 'remote.origin.promisor', 'true'])
        run(['git', '-C', str(dst), 'config', 'remote.origin.partialclonefilter', 'blob:none'])
        # Do not reset, clean or overwrite an existing populated source checkout.
        head = subprocess.run(['git', '-C', str(dst), 'rev-parse', '--verify', 'HEAD'], text=True, capture_output=True)
        if head.returncode == 0 and head.stdout.strip() != entry['commit']:
            raise ValueError('Existing source points to a different commit')
        if head.returncode != 0:
            run(['git', '-C', str(dst), 'fetch', '--depth=1', '--filter=blob:none', '--no-tags', 'origin', entry['commit']])
        # Materialize only verified text blobs before checkout. Git's lazy fetch
        # can otherwise issue one network request per file or stall on old Git.
        def materialize(item):
            relative, info = item
            p = dst / relative
            if p.is_symlink() or Path(relative).is_absolute() or '..' in Path(relative).parts:
                raise ValueError('Unsafe source file')
            if p.exists():
                data = p.read_bytes()
            else:
                repo = entry['repository'].removeprefix('https://github.com/').removesuffix('.git')
                url = 'https://raw.githubusercontent.com/' + repo + '/' + entry['commit'] + '/' + urllib.parse.quote(relative)
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if blob != info['git_blob']:
                raise ValueError('Changed existing or downloaded source: ' + relative)
            subprocess.run(['git', '-C', str(dst), 'hash-object', '-w', '--stdin'], input=data, check=True, capture_output=True)
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(materialize, entry['files'].items()))
        if head.returncode != 0:
            run(['git', '-C', str(dst), 'sparse-checkout', 'init', '--no-cone'])
            names = []
            for path in entry['files']:
                if Path(path).is_absolute() or '..' in Path(path).parts or '\n' in path:
                    raise ValueError('Unsafe source path')
                names.append('/' + path)
            run(['git', '-C', str(dst), 'sparse-checkout', 'set', '--no-cone', '--stdin'], input='\n'.join(names) + '\n')
            run(['git', '-C', str(dst), 'checkout', '--detach', entry['commit']])
        if run(['git', '-C', str(dst), 'status', '--porcelain']):
            raise ValueError('Source checkout contains changes outside the pinned source inventory')
        for relative, info in entry['files'].items():
            p = dst / relative
            if p.is_symlink():
                raise ValueError('Symlink source not accepted')
            data = p.read_bytes()
            actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if actual != info['git_blob']:
                raise ValueError('Source blob mismatch: ' + str(p))
        receipt = {'id': entry['id'], 'commit': entry['commit'], 'repository': entry['repository'],
                   'verified_files': len(entry['files']), 'path': str(dst.resolve())}
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
    result = {'status': 'SOURCE_CLONES_VERIFIED', 'repositories': receipts,
              'upstream_programs_executed': False, 'model_weights_requested': False}
    (root / 'CLONES_VERIFIED.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ids', nargs='+')
    args = parser.parse_args()
    entries = json.loads((HERE / 'sources.lock.json').read_text())['repositories']
    if args.ids:
        if not set(args.ids) <= {e['id'] for e in entries}:
            parser.error('Unknown source id')
        entries = [e for e in entries if e['id'] in args.ids]
    clone(args.root.resolve(), entries)
