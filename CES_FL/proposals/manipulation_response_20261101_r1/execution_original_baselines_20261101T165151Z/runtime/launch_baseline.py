"""Launch exactly one dry-run OR training arm; no automatic milestone chaining."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def gate(arm, phase, host, device_uuid):
    if arm not in ('bce', 'fair') or phase not in ('dry', 'train') or host not in ('gpu01', 'gpu02'):
        raise ValueError('Only the approved original-only baseline milestones are supported')
    verified = json.loads((ROOT / 'checks/P_OPERATOR_VERIFIED.json').read_text())
    if verified['status'] != 'PASS' or verified['completed_epochs'] != 50:
        raise ValueError('P50 has not passed operator verification')
    bound_path = ROOT / 'bound_originals/bound_config.json'
    bound = json.loads(bound_path.read_text())
    if bound['bank'] is not None or bound['pretrain_sha256'] != verified['checkpoint_sha256']:
        raise ValueError('Original-only binding mismatch')
    if phase == 'train':
        dry = json.loads((ROOT / ('dry_' + arm) / 'DRY_RUN.json').read_text())
        passed = json.loads((ROOT / 'checks' / ('dry_' + arm + '_PASS.json')).read_text())
        binding = passed['binding']
        if not (dry['status'] == passed['status'] == 'PASS' and dry['arm'] == arm
                and dry['device'] == 'cuda' and dry['optimizer_steps'] == 0
                and dry['quartets'] == 0 and dry['synthetic_fixture'] is False):
            raise ValueError('Required configured baseline CUDA backward check failed')
        if not all(math.isfinite(v) for v in dry['losses'].values()):
            raise ValueError('Nonfinite recorded backward objective')
        if binding['host'] != host or binding['gpu_uuid'] != device_uuid:
            raise ValueError('Train must use the same actual device/host as its backward check')
        if binding['bound_config_sha256'] != sha(bound_path):
            raise ValueError('Bound configuration changed after backward check')
        if binding['pretrain_checkpoint_sha256'] != bound['pretrain_sha256']:
            raise ValueError('Pretraining dependency changed after backward check')
        if (ROOT / arm).exists():
            raise ValueError('Existing training outputs must be preserved')
    return bound


def main(arm, phase, host, device_uuid):
    gate(arm, phase, host, device_uuid)
    stage = 'dry_' + arm if phase == 'dry' else arm
    if (ROOT / 'checks' / (stage + '_SUBMISSION.json')).exists():
        raise ValueError('Submission already recorded; do not duplicate')
    script = 'baseline_dry_worker.py' if phase == 'dry' else 'direct_mr.py'
    remote = """import json,os,socket,subprocess
from pathlib import Path
r=Path(ROOT_VALUE); stage=STAGE_VALUE; script=SCRIPT_VALUE; device_uuid=DEVICE_VALUE
assert not (r/'checks'/(stage+'_SUBMISSION.json')).exists()
assert not (r/'checks'/(stage+'_PASS.json')).exists()
assert not (r/'checks'/(stage+'_FAILURE.json')).exists()
argv=['/home/sachin.chaudhary/.conda/envs/aiface/bin/python','-u',str(r/'runtime'/script),'--stage',stage]
environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
if device_uuid: environment['CUDA_VISIBLE_DEVICES']=device_uuid
with (r/'logs'/(stage+'_worker.log')).open('x') as log:
 p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
result={'pid':p.pid,'host':socket.gethostname().split('.')[0],'stage':stage,'command':argv,'execution_mode':'direct_ssh_bounded_shared_gpu','pbs_allocation':None,'restricted_device_uuid':device_uuid}
with (r/'checks'/(stage+'_SUBMISSION.json')).open('x') as f: json.dump(result,f,indent=2)
print(json.dumps(result))
""".replace('ROOT_VALUE', repr(str(ROOT))).replace('STAGE_VALUE', repr(stage)).replace(
        'SCRIPT_VALUE', repr(script)).replace('DEVICE_VALUE', repr(device_uuid))
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', host,
               "python - <<'BASELINE'\n" + remote + '\nBASELINE']
    result = {'command': command, 'stage': stage, 'phase': phase, 'arm': arm,
              'observed_management_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'Single original-only baseline milestone; no test/paired data or next-stage launch'}
    with (ROOT / 'checks' / (stage + '_LAUNCH_ATTEMPT.json')).open('x') as f:
        json.dump(result, f, indent=2)
    p = subprocess.run(command, capture_output=True, text=True, timeout=35)
    result.update(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
    (ROOT / 'checks' / (stage + '_LAUNCH_RECEIPT.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if p.returncode:
        raise RuntimeError('Ambiguous/failed SSH submission: preserve and reconcile before recovery')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['bce', 'fair'], required=True)
    p.add_argument('--phase', choices=['dry', 'train'], required=True)
    p.add_argument('--host', choices=['gpu01', 'gpu02'], required=True)
    p.add_argument('--device-uuid')
    a = p.parse_args(); main(a.arm, a.phase, a.host, a.device_uuid)
