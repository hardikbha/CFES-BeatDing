"""Observe only this authorized batch; publish one dedicated GitHub status page.

No job submission, cancellation, model/config changes, threshold fitting or
automatic experiment retries. Stops when all phases are terminal or after24h.
"""
import argparse
import base64
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time


def read(path):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return None


def observe(root):
    jobs = read(root/'JOBS.json'); rows = []
    for phase, job in jobs['jobs'].items():
        state = 'UNVERIFIED'; exit_status = None
        try:
            p = subprocess.run(['qstat', '-xf', job['job_id']], capture_output=True, text=True, timeout=15)
            for line in p.stdout.splitlines():
                if line.strip().startswith('job_state ='): state = line.split('=', 1)[1].strip()
                if line.strip().startswith('Exit_status ='): exit_status = int(line.split('=', 1)[1])
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass  # Missing scheduler telemetry is never proof of training/completion.
        progress = read(root/'runtime'/f'{phase}.progress.json')
        result = read(root/'runtime'/f'{phase}.result.json')
        failure = read(root/'runtime'/f'{phase}.failure.json')
        if state in ('F', 'C') and exit_status not in (None, 0) and failure is None:
            failure = {'pbs_exit_status': exit_status}
        status = 'DONE' if result and result.get('status') == 'DONE' else 'FAILED' if failure else progress.get('phase', 'QUEUED') if progress else 'QUEUED' if state in ('Q', 'H', 'W') else 'STARTING_OR_UNVERIFIED'
        rows.append({'phase': phase, 'job_id': job['job_id'], 'pbs_state': state, 'state': status,
                     'progress': progress, 'failure': failure})
    for row in rows:
        dependency = jobs['jobs'][row['phase']].get('dependency_phase')
        if dependency and any(r['phase'] == dependency and r['state'] == 'FAILED' for r in rows):
            row['state'] = 'BLOCKED_UPSTREAM_FAILURE'
    metrics = []
    for path in sorted((root/'evaluation').glob('*/*/metrics.json')):
        m = read(path)
        metrics.append({'arm': path.parent.parent.name, 'dataset': path.parent.name,
                        'auc': m['calibrated']['overall']['auc'], 'actual_fpr': m['calibrated']['overall']['fpr'],
                        'intersection': m['calibrated']['axes']['gender_race']['ding_style_fractions'],
                        'finished': (path.parent/'DONE').is_file()})
    return {'observed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'jobs': rows, 'metrics': metrics,
            'source_commit': jobs['source_commit'], 'run_root': str(root), 'request_id': jobs['request_id']}


def render(s):
    lines = ['# Authorized seed17 benchmark — live batch observations', '',
             'Observed UTC: **'+s['observed_at_utc']+'**. Dedicated monitor publishes at state changes and periodic heartbeats; last push is still a snapshot.',
             '', 'Pinned source commit: `'+s['source_commit']+'`.', '',
             '| Phase | PBS job | PBS state | Observed work state |', '|---|---|---|---|']
    for r in s['jobs']:
        lines.append('| '+' | '.join([r['phase'], r['job_id'], r['pbs_state'], r['state']])+' |')
    lines += ['', 'Q means queued; R alone does not mean model training has begun. WAITING_FOR_EMPTY_GPU means a PBS allocation exists but the conservative physical-device lease is waiting. No other job is stopped.', '',
              'ResNet B/C depend on successful ResNet P. They use exactly its selected encoder, identical fresh heads, fixed validation-AUC selection and stored validation-real thresholds. C remains invariance-only on36reviewed edits/six TRAIN sources. No new image generation or Ding reproduction.', '',
              '## Actual test results appearing so far', '',
              '| Arm | Dataset | AUC (%) | Actual test FPR (%) | Intersection F_FPR (%) | Intersection es-AUC (%) | Inference + CI finished |',
              '|---|---|---:|---:|---:|---:|---|']
    def number(v): return f'{100*v:.2f}' if v is not None else 'undefined'
    for m in s['metrics']:
        i = m['intersection'] or {}
        lines.append('| '+' | '.join([m['arm'], m['dataset'], number(m['auc']), number(m['actual_fpr']),
                     number(i.get('F_FPR')), number(i.get('es_AUC')), str(m['finished'])])+' |')
    if not s['metrics']: lines += ['', 'No new test scores yet; jobs may still be queued or running their pre-test checks.']
    lines += ['', 'Thresholds are frozen from each checkpoint’s validation calibration; achieved test FPR is measured, not guaranteed. Separate fixed-probability0.5 reports and all group counts are in the result files. Native DFDC contains1,476exact duplicate rows; their original weighting is preserved and explicitly reported. Some native subgroups have only one class; undefined metrics are not fabricated.', '',
              'FairForge-2 here is cross-dataset evaluation of FF++-trained models, not an in-domain FairForge-2 training claim. Bootstrap uses connected recorded-contributor components; if all rows are connected, no independent-component CI is claimed. Proxy identities, hidden donors and a single seed limit inference. These are exploratory comparisons with documented protocol differences from Ding.', '',
              'Cluster artifacts: `'+s['run_root']+'`.', '',
              'This monitor observes/publishes only the submitted batch and stops after completion/failure or24h. It never submits more jobs, retries training, changes code/configuration, selects on test or terminates jobs. Historical CESel-MA gate remains FAIL.']
    return '\n'.join(lines)+'\n'


def publish(request_id, body):
    route = 'repos/hardikbha/CFES-BeatDing/contents/CES_FL/status/'+request_id+'_live.md'
    current = subprocess.run(['gh', 'api', route], capture_output=True, text=True, timeout=30)
    payload = {'message': 'Update '+request_id+' benchmark progress', 'branch': 'main',
               'content': base64.b64encode(body.encode()).decode()}
    if current.returncode == 0: payload['sha'] = json.loads(current.stdout)['sha']
    p = subprocess.run(['gh', 'api', '--method', 'PUT', route, '--input', '-'], input=json.dumps(payload), capture_output=True, text=True, timeout=45)
    if p.returncode: raise RuntimeError(p.stderr[-1000:])
    return json.loads(p.stdout)['commit']['sha']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--once', action='store_true'); p.add_argument('--publish', action='store_true'); args = p.parse_args()
    root = args.run_root; guard = (root/'runtime/monitor.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time(); previous = None; last_publish = 0
    while True:
        s = observe(root); body = render(s)
        (root/'LIVE_STATUS.md').write_text(body)
        (root/'runtime/live_snapshot.json').write_text(json.dumps(s, indent=2)+'\n')
        milestones = [{k: j[k] for k in ('phase', 'job_id', 'pbs_state', 'state')} for j in s['jobs']]
        signature = hashlib.sha256(json.dumps({'jobs': milestones, 'metrics': s['metrics']}, sort_keys=True).encode()).hexdigest()
        if args.publish and (signature != previous or time.time()-last_publish > 1800):
            try:
                commit = publish(s['request_id'], body)
                print('GITHUB_PROGRESS_PUSH', commit, s['observed_at_utc'], flush=True)
                previous = signature; last_publish = time.time()
            except Exception as e:
                print('GITHUB_PROGRESS_ERROR', repr(e), flush=True)
        terminal = all(j['state'] in ('DONE', 'FAILED', 'BLOCKED_UPSTREAM_FAILURE') for j in s['jobs'])
        if args.once or terminal or time.time()-started > 86400: break
        time.sleep(30)


if __name__ == '__main__': main()
