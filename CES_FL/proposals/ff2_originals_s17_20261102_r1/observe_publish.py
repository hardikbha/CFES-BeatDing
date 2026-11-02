"""Finite read-only FF2 observer: publish milestones; never launch or control GPU work."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1')
TARGET = 'CES_FL/status/ff2_originals_s17_20261102_r1_live.md'


def git(worktree, *args):
    return subprocess.check_output(['git', '-c', 'gc.auto=0', '-C', str(worktree), *args], text=True).strip()


def observe():
    receipt = json.loads((ROOT / 'checks/train_CHILD.json').read_text())
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', receipt['host'],
                        'ps', '-p', str(receipt['child_pid']), '-o', 'pid=,etime=,args='],
                       capture_output=True, text=True, timeout=15)
    alive = p.returncode == 0 and bool(p.stdout.strip())
    history = []
    hp = ROOT / 'training/history.jsonl'
    if hp.exists():
        for line in hp.read_text().splitlines():
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                break  # An actively appended partial line is not a completed epoch.
    verified = ROOT / 'checks/train_PASS.json'
    failure = ROOT / 'checks/train_FAILURE.json'
    status = ('COMPLETED' if verified.exists() else 'FAILED' if failure.exists() else
              'RUNNING' if alive and (ROOT / 'training/provenance.json').exists() else
              'STARTING_INPUT_LOADING' if alive else 'UNVERIFIED_PROCESS_NOT_FOUND')
    selected = max(history, key=lambda h: h['validation']['overall']['auc']) if history else None
    key = (status, len(history))
    lines = ['# FairForge-2 original-only CES-FL pretraining — seed17\n',
             'Observed management UTC: ' + datetime.datetime.now(datetime.timezone.utc).isoformat() + '.\n',
             f"Status: **{status}**. Completed epochs: **{len(history)}/10**.\n",
             f"Direct SSH host **{receipt['host']}**, physical GPU **{receipt['physical_index']}**, H10080GB; trainer PID **{receipt['child_pid']}**, worker PID **{receipt['worker_pid']}**. GPU UUID `{receipt['gpu_uuid']}`. No PBS allocation is claimed; cooperative locks and20% VRAM cap. Existing jobs preserved.\n",
             'Actual child process observation: `' + p.stdout.strip().replace('`', '') + '`.\n',
             'TRAIN218,266; VAL31,107. TEST87,564 is not evaluated. Original-image BCE; all21,594,159parameters trainable, BNupdates. Validation AUC selects checkpoints; calibration uses validation reals only.\n',
             'Input hash/reader/path/provenance checks PASS;64TRAIN+64VALimages decoded by stratum. Native CUDA suite and two real TRAIN-only optimizer steps PASS with losses0.71599847,0.70625442. No synthetic edit is used.\n',
             '| Epoch | Validation AUC (%) | Classification loss |', '|---|---:|---:|']
    for h in history:
        lines.append(f"| {h['epoch']} | {100*h['validation']['overall']['auc']:.6f} | {h['losses']['classification']:.8f} |")
    if selected:
        report = selected['validation']; calibration = selected['calibration']
        lines += [f"\nCurrent validation-selected epoch: {selected['epoch']}. Achieved VALFPR: {100*calibration['achieved_fpr']:.6f}% ({calibration['achieved_fp']}/{calibration['n_validation_real']}); no TESTFPR guarantee.\n",
                  '| Attribute | F_FPR (%) | F_DP (%) | es-AUC (%) | Worst eligible AUC (%) |', '|---|---:|---:|---:|---:|']
        for axis in ['gender', 'race', 'gender_race', 'age_bin', 'skin_bin']:
            a = report['axes'][axis]; d = a['ding_style_fractions']
            if d is not None:
                vals = [100*d[k] for k in ['F_FPR', 'F_DP', 'es_AUC']] + [100*a['worst_group_auc']]
                lines.append('| ' + axis + ' | ' + ' | '.join(f'{v:.6f}' for v in vals) + ' |')
    if verified.exists():
        v = json.loads(verified.read_text())
        lines += [f"\nSelected checkpoint SHA256: `{v['checkpoint_sha256']}`. Native independent selected-checkpoint VAL replay and strict reload PASS.\n"]
    if failure.exists():
        lines.append('\nActual failure: `' + json.loads(failure.read_text())['error'].replace('`', '') + '`. No automatic retry.\n')
    lines += ['\nModel source commit `875bce025084871438ef799ca9086985c4375672`; exact fixed FF2P10config SHA256 `db0c886527635295f0394dddbca9da1a8a3c40807609ab78f324b75b382fbee8`. No architecture/loss edits or FF++configuration substitution.\n',
              'Run root: `/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1`; actual training log: `logs/train.log`; history/checkpoints: `training/`; checks: `checks/`.\n',
              '[Exact request, configuration, fresh checks and scope](../proposals/ff2_originals_s17_20261102_r1/README.md). Full matched Ding comparison, paired arms and newTEST evaluation remain blocked/unlaunched. Current run is original-only P, not a completed counterfactual comparison.\n',
              'This finite read-only observer publishes state/epoch milestones and stops after completion, failure, process ambiguity or12h. It cannot launch training, generate images, change selection/calibration or kill jobs. Timestamps remain observations, not a monitoring guarantee.\n']
    return key, '\n'.join(lines)


def main(worktree):
    deadline = time.monotonic() + 12*3600
    previous = None
    while time.monotonic() < deadline:
        key, text = observe()
        if key != previous:
            assert not git(worktree, 'status', '--porcelain')
            git(worktree, 'fetch', 'origin', 'main')
            git(worktree, 'rebase', 'origin/main')
            (worktree / TARGET).write_text(text)
            git(worktree, 'add', '--', TARGET)
            git(worktree, 'diff', '--cached', '--check')
            git(worktree, 'commit', '-m', f'Observe FF2 P seed17 {key[0]} epoch {key[1]}/10')
            git(worktree, 'push', 'origin', 'HEAD:main')
            print('FF2_OBSERVATION_PUBLISHED', key, git(worktree, 'rev-parse', 'HEAD'), flush=True)
            previous = key
        if key[0] in ['COMPLETED', 'FAILED', 'UNVERIFIED_PROCESS_NOT_FOUND']:
            print('OBSERVER_STOP', key, flush=True); return
        time.sleep(30)
    print('OBSERVER_TIME_LIMIT_REACHED_NO_GPU_ACTION', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--publish-worktree', type=Path, required=True)
    main(p.parse_args().publish_worktree)
