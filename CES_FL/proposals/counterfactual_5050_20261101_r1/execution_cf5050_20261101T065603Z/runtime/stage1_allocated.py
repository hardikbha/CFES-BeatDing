"""Allocation-guarded milestone 1 only; never launches pilot/full training."""
import datetime
import getpass
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-cf5050-fc4e63b')
COMMIT = 'fc4e63b8811c5813afa0d298a8172ba9555b7e40'
SCRIPT = SOURCE / 'CES_FL/src/ces_fl_counterfactual_5050.py'
CONFIG = SOURCE / 'CES_FL/proposals/counterfactual_5050_20261101_r1/protocol_config.json'


def write(name, data):
    path = ROOT / 'checks' / name
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def allocation():
    jobid = os.environ.get('PBS_JOBID')
    if not jobid:
        raise RuntimeError('NO_PBS_ALLOCATION: milestone 1 requires a real PBS GPU job')
    raw = subprocess.check_output(['/opt/pbs/bin/qstat', '-f', '-F', 'json', jobid], text=True)
    all_jobs = json.loads(raw)['Jobs']
    job = all_jobs.get(jobid) or next(iter(all_jobs.values()))
    host = socket.gethostname().split('.')[0]
    owner = job['Job_Owner'].split('@')[0]
    if owner != getpass.getuser() or job['job_state'] != 'R':
        raise RuntimeError('Wrong owner or non-running allocation')
    resources = job['Resource_List']
    if int(resources.get('ngpus', 0)) != 1 or job['queue'] != 'gpu':
        raise RuntimeError('Exactly one allocated GPU in the gpu queue is required')
    vnode = job.get('exec_vnode', '')
    if not re.search(r'\(' + re.escape(host) + r':[^)]*\bngpus=1\b', vnode):
        raise RuntimeError('Host/GPU request does not match actual exec_vnode')
    record = {'job_id': jobid, 'host': host, 'owner': owner, 'job_state': job['job_state'],
              'queue': job['queue'], 'Resource_List': resources, 'exec_vnode': vnode,
              'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    # No CUDA binding was supplied at submission. Accept only an actual runtime
    # scheduler binding; a free physical device is not evidence of ownership.
    binding = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if binding:
        if len(binding.split(',')) != 1 or binding in ('-1', 'all'):
            raise RuntimeError('Scheduler CUDA_VISIBLE_DEVICES is not a single device')
        record['binding_source'] = 'runtime scheduler CUDA_VISIBLE_DEVICES; none supplied to qsub'
    else:
        gpu_file = os.environ.get('PBS_GPUFILE')
        entries = Path(gpu_file).read_text().splitlines() if gpu_file and Path(gpu_file).is_file() else []
        entries = [x.strip() for x in entries if x.strip()]
        record['PBS_GPUFILE'] = gpu_file
        record['gpu_file_entries'] = entries
        if len(entries) == 1:
            match = re.fullmatch(r'([A-Za-z0-9_.-]+)-(?:gpu|nvidia)([0-9]+)', entries[0])
            if match and match[1].split('.')[0] == host:
                binding = match[2]
                os.environ['CUDA_VISIBLE_DEVICES'] = binding
                record['binding_source'] = 'PBS_GPUFILE assigned device'
        if not binding:
            record['status'] = 'GPU_BINDING_UNVERIFIED'
            write('ALLOCATION.json', record)
            raise RuntimeError('GPU_BINDING_UNVERIFIED: no scheduler CUDA_VISIBLE_DEVICES or usable PBS_GPUFILE; refusing to guess a device')
    record['cuda_visible_devices'] = binding
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('CUDA is unavailable or the allocated binding exposes more than one device')
    props = torch.cuda.get_device_properties(0)
    record.update(status='PASS', device_name=props.name, total_vram_bytes=props.total_memory,
                  torch=str(torch.__version__), cuda_version=torch.version.cuda,
                  device_uuid=str(getattr(props, 'uuid', 'UNAVAILABLE')))
    free, total = torch.cuda.mem_get_info()
    record.update(free_vram_bytes=free, physical_vram_bytes=total)
    write('ALLOCATION.json', record)
    return record


def run():
    baseline = json.loads((ROOT / 'checks/LOCAL_HASH_PREFLIGHT.json').read_text())
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True)
    assert sha(SCRIPT) == baseline['source_sha256'] and sha(CONFIG) == baseline['config_sha256']
    for value in baseline['inputs'].values():
        assert sha(Path(value['path'])) == value['sha256'], 'Declared input changed'
    assigned = allocation()
    commands = [
        ('smoke_cuda', [sys.executable, '-u', str(SCRIPT), 'smoke', '--device', 'cuda', '--out', str(ROOT / 'smoke_cuda')]),
        ('generation_plan', [sys.executable, '-u', str(SCRIPT), 'plan', '--config', str(CONFIG), '--out', str(ROOT / 'generation')]),
        ('dry_pretrain', [sys.executable, '-u', str(SCRIPT), 'dry-run', '--config', str(CONFIG), '--stage', 'pretrain', '--device', 'cuda', '--out', str(ROOT / 'dry_pretrain')]),
    ]
    write('STAGE1_COMMANDS.json', {'job_id': assigned['job_id'], 'commands': commands,
                                  'pilot_or_full_training_included': False})
    for name, command in commands:
        print('START', name, json.dumps(command), flush=True)
        with (ROOT / 'logs' / (name + '.log')).open('x') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=SOURCE)
        write(name + '_EXIT.json', {'command': command, 'exit_code': completed.returncode,
                                   'job_id': assigned['job_id'], 'log': str(ROOT / 'logs' / (name + '.log'))})
        if completed.returncode:
            raise RuntimeError(name + ' failed; all later milestones remain stopped')
        print('PASS', name, flush=True)
    smoke = json.loads((ROOT / 'smoke_cuda/RESULT.json').read_text())
    dry = json.loads((ROOT / 'dry_pretrain/RESULT.json').read_text())
    # The source currently writes its smoke result as RESULT.json. If that
    # contract changes, stop rather than claim a software PASS from an exit code.
    assert smoke['status'] == 'PASS' and smoke['device'] == 'cuda'
    assert dry['status'] == 'PASS' and dry['optimizer_steps'] == 0 and not dry['synthetic_fixture']
    write('STAGE1_PASS.json', {'status': 'PASS', 'source_commit': COMMIT, 'job_id': assigned['job_id'],
                              'cuda_smoke': smoke, 'real_data_dry_run': dry,
                              'generation_plan': str(ROOT / 'generation'), 'generation_started': False,
                              'full_training_started': False, 'next_milestone': 'Separate two-source Qwen pilot; no automatic submission'})


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        write('STAGE1_FAILURE.json', {'status': 'FAIL_OR_BLOCKED', 'exception': str(exc),
                                     'traceback': traceback.format_exc(), 'job_id': os.environ.get('PBS_JOBID'),
                                     'full_training_started': False, 'generation_started': False})
        raise
