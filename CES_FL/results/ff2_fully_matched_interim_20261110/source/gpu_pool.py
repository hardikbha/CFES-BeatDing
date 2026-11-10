"""One bounded study arm, on an explicitly leased shared H100. No TEST."""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parent
ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/ff2_fully_matched_20261110_r1')
LOCKS = Path('/tmp/scf_gpu_locks')
PYTHON = '/home/sachin.chaudhary/.conda/envs/aiface/bin/python'
NEED = 16000
ARM_NAMES = ('A_BCE', 'F_fairness', 'R_response', 'H_full')

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(p)

def alive(pid, group=False):
    if not isinstance(pid, int) or pid <= 0: return False
    try:
        (os.killpg if group else os.kill)(pid, 0)
        return True
    except ProcessLookupError: return False
    except PermissionError: return True

@contextlib.contextmanager
def mutex():
    LOCKS.mkdir(exist_ok=True)
    lock = LOCKS / '.pool_mutex'
    deadline = time.monotonic() + 30
    while True:
        try: lock.mkdir(); break
        except FileExistsError:
            if time.monotonic() > deadline: raise RuntimeError('Shared pool mutex busy')
            time.sleep(.2)
    try: yield
    finally: lock.rmdir()

def candidates():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    cards = [dict(index=int(r[0]), uuid=r[1].strip(), free_mib=int(r[2]), util=int(r[3]))
             for r in csv.reader(text.splitlines())]
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    apps = []
    for row in csv.reader(text.splitlines()):
        try: apps.append((row[0].strip(), os.getpgid(int(row[1])), int(row[2])))
        except (ValueError, IndexError, ProcessLookupError, PermissionError): pass
    result = []
    for card in cards:
        directory = LOCKS / ('gpu' + str(card['index']))
        entries = []
        if directory.exists():
            try:
                if not read(directory / 'pool_owner.json').get('shared'): continue
                entries = [read(p) for p in directory.glob('task_*.json')]
                entries = [e for e in entries if alive(e.get('pid')) or alive(e.get('pgid'), True)]
            except (OSError, ValueError, TypeError): continue
        # One new arm per card, and do not share with a reserved Ding study.
        if any(str(e.get('name', '')).startswith(('ff2suite_', 'dingmatch_', 'lhstab_', 'ding_ff2_public')) for e in entries): continue
        reserve = sum(max(0, int(e.get('need', 0)) - sum(m for u, pg, m in apps
                          if u == card['uuid'] and pg == e.get('pgid'))) for e in entries)
        card['available_mib'] = card['free_mib'] - reserve
        if card['available_mib'] >= NEED + 4096 and card['util'] <= 15: result.append(card)
    return sorted(result, key=lambda c: (-c['available_mib'], c['util']))

def verify_source():
    record = read(ROOT / 'SOURCE_CHECKOUT.json')
    head = (SOURCE / '.git/HEAD').read_text().strip()
    if head != record['commit'] or len(head) != 40: raise RuntimeError('Detached pinned HEAD required')
    for relative, expected in record['source_files'].items():
        if sha(SOURCE / relative) != expected: raise RuntimeError('Source changed: ' + relative)
    return record

def claim(name):
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        with mutex():
            options = candidates()
            if options:
                card = options[0]
                directory = LOCKS / ('gpu' + str(card['index']))
                if not directory.exists():
                    directory.mkdir()
                    write(directory / 'pool_owner.json', dict(shared=True, v=2, pid=0, wid='pool_v2_shared'))
                entry = directory / ('task_ff2suite_' + name + '.json')
                if entry.exists(): raise RuntimeError('Own lease name exists; preserve it')
                write(entry, dict(name='ff2suite_' + name, wid='ff2suite_' + name, pid=os.getpid(),
                                  pgid=None, cls='reader', need=NEED, time=time.time()))
                return card, entry
        write(ROOT / name / 'LIVE.json', dict(status='WAITING_FOR_SHARED_GPU', host=os.uname().nodename,
                                             pid=os.getpid(), observed_unix=time.time()))
        time.sleep(15)
    raise RuntimeError('No eligible idle shared device in two hours; no preemption')
