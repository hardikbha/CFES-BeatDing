"""CUDA smoke and 20-step real-TRAIN profiles for locally weighted FF2 adapters."""
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path('/home/sachin.chaudhary/ces_fl_sources/ff2-expanded-e7648fb')
ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/ff2_expanded_e7648fb')
SOURCES = Path('/home/sachin.chaudhary/ces_fl_dependencies/ff2_expanded_e7648fb_sources/aiface')
PY = '/home/sachin.chaudhary/.conda/envs/aiface/bin/python'
METHODS = ('efficientnetb4', 'f3net', 'spsl', 'srm', 'core', 'univfd')
session = Path(sys.argv[1]).resolve()
runner = SOURCE / 'CES_FL/src/ff2_expanded_runner.py'


def run(args):
    print('COMMAND', json.dumps([str(x) for x in args]), flush=True)
    subprocess.run([str(x) for x in args], cwd=SOURCE, check=True)


for method in METHODS:
    config = ROOT / f'bound_configs/{method}.json'
    preflight = ROOT / f'preflight_reused_v2/{method}'
    output = session / method
    output.mkdir()
    run([PY, runner, 'smoke', '--method', method, '--sources', SOURCES,
         '--device', 'cuda', '--out', output / 'cuda_smoke'])
    run([PY, runner, 'dry-run', '--config', config, '--sources', SOURCES,
         '--preflight', preflight, '--device', 'cuda', '--steps', '20',
         '--out', output / 'real_train_profile'])
print('ALL_PROFILES_PASS', flush=True)
