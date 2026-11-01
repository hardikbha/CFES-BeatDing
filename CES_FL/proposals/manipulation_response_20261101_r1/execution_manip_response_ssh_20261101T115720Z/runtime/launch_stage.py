"""Submit one explicitly requested SSH milestone; never schedules the next one."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z')


def main(stage, host):
    assert stage in ('real_preflight', 'P', 'bce', 'fair') and host in ('gpu01', 'gpu02')
    if stage == 'real_preflight':
        assert json.loads((ROOT / 'checks/smoke_PASS.json').read_text())['status'] == 'PASS'
    else:
        assert json.loads((ROOT / 'checks/real_preflight_PASS.json').read_text())['status'] == 'PASS'
    assert not (ROOT / 'checks' / (stage + '_SUBMISSION.json')).exists()
    remote = """import json,os,socket,subprocess
from pathlib import Path
r=Path(ROOT_VALUE)
stage=STAGE_VALUE
assert not (r/'checks'/(stage+'_SUBMISSION.json')).exists()
assert not (r/'checks'/(stage+'_PASS.json')).exists()
assert not (r/'checks'/(stage+'_FAILURE.json')).exists()
argv=['/home/sachin.chaudhary/.conda/envs/aiface/bin/python','-u',str(r/'runtime/direct_mr.py'),'--stage',stage]
with (r/'logs'/(stage+'_worker.log')).open('x') as log:
 p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
d={'pid':p.pid,'host':socket.gethostname().split('.')[0],'stage':stage,'command':argv,'execution_mode':'direct_ssh_bounded_shared_gpu','pbs_allocation':None}
with (r/'checks'/(stage+'_SUBMISSION.json')).open('x') as f:json.dump(d,f,indent=2)
print(json.dumps(d))
""".replace('ROOT_VALUE', repr(str(ROOT))).replace('STAGE_VALUE', repr(stage))
    argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', host,
            "python - <<'MRSTAGE'\n" + remote + '\nMRSTAGE']
    record = {'command': argv, 'stage': stage, 'host': host,
              'observed_management_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with (ROOT / 'checks' / (stage + '_LAUNCH_ATTEMPT.json')).open('x') as f:
        json.dump(record, f, indent=2)
    p = subprocess.run(argv, capture_output=True, text=True, timeout=35)
    record.update(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
    (ROOT / 'checks' / (stage + '_LAUNCH_RECEIPT.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)
    if p.returncode:
        raise RuntimeError('SSH launch failed/ambiguous; preserve receipts and reconcile before any recovery')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['real_preflight', 'P', 'bce', 'fair'], required=True)
    p.add_argument('--host', choices=['gpu01', 'gpu02'], required=True)
    a = p.parse_args(); main(a.stage, a.host)
