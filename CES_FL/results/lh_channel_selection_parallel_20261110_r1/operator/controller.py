"""Finite operator for the pinned three-arm pilot; no scientific-source edits."""
import contextlib
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

CONTROL = Path(__file__).resolve().parent
SCIENCE = Path('/home/sachin.chaudhary/ces_fl_sources/lh-channel-selection-1a6d63db6baf-20261110')
SCRIPT = SCIENCE / 'CES_FL/src/ces_fl_lh_channel_selection.py'
ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/lh_channel_selection_20261110_r1_operator_1a6d63d')
PYTHON = '/home/sachin.chaudhary/.conda/envs/aiface/bin/python'
BUDGET_HELPER = Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/allocated_stage.py')
POOL_HELPER = Path('/home/sachin.chaudhary/ces_fl_sources/ff2-fully-matched-20261110/gpu_pool.py')
SESSION = 'lh_channel_selection_1a6d63d_parallel_20261110_r1'
ARMS = ('none', 'random', 'informed')
GPU_GUARD_SECONDS = 6840  # Retain six aggregate GPU-minutes for bounded cleanup.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detached_head(root):
    root = Path(root)
    meta = root / '.git'
    if meta.is_file():
        text = meta.read_text().strip()
        if not text.startswith('gitdir: '):raise ValueError('Unknown worktree metadata')
        target = Path(text[len('gitdir: '):])
        meta = target if target.is_absolute() else root / target
    head = (meta / 'HEAD').read_text().strip()
    if len(head) != 40 or any(c not in '0123456789abcdef' for c in head):
        raise ValueError('A detached pinned HEAD is required')
    return head


def held_seconds(leases, now):
    return sum(max(0., (x['ended'] if x.get('ended') is not None else now) - x['started']) for x in leases)


def cost_gate(per_arm, used, setup_margin=60.):
    return (set(per_arm) == set(ARMS)
            and all(math.isfinite(v) and 0 < v <= 1740 for v in per_arm.values())
            and used + sum(per_arm.values()) + setup_margin < GPU_GUARD_SECONDS)


def verify():
    binding = read(ROOT / 'PARALLEL_CONTROLLER_BINDING.json')
    assert detached_head(CONTROL) == binding['control_commit']
    for name, expected in binding['control_files'].items():
        assert sha(CONTROL / name) == expected, name
    assert detached_head(SCIENCE) == '1a6d63db6baf7f6ad2a8e842a31341d2c128de25'
    for name,expected in binding['science_files'].items():
        assert sha(SCIENCE/name) == expected,name
    assert sha(SCRIPT) == '95574295b76f16f5e5d8f70f01502848b4616fc7437030a5dfdf2ff1219f62f0'
    assert sha(BUDGET_HELPER) == binding['budget_helper_sha256']
    assert sha(POOL_HELPER) == binding['pool_helper_sha256']
    q = read(ROOT / 'QUALIFIED.json')
    assert q['status'] == 'OPERATOR_REVIEW_RECORDED' and q['case_count'] == 344
    assert sha(q['review_file']) == q['review_sha256'] and sha(q['preview_file']) == q['preview_sha256']
    assert q['source_sha256'] == sha(SCRIPT)
    checks = read(Path(binding['CPU_checks_file']))
    assert checks['status'] == 'PASS' and len(checks['checks']) == 5
    assert sha(binding['controller_CPU_tests_file']) == binding['controller_CPU_tests_sha256']
    override = read(ROOT / 'USER_PARALLEL_SCHEDULING_OVERRIDE.json')
    assert override['maximum_parallel_configuration_count'] == 3
    assert override['maximum_aggregate_GPU_hours'] == 2 and override['epochs_per_arm'] == 3
    return binding


class Operator:
    def __init__(self):
        self.binding = verify()
        self.budget = load(BUDGET_HELPER, 'lh_budget')
        self.pool = load(POOL_HELPER, 'lh_pool')
        self.leases = []
        self.children = []
        self.started = None
        self.start_utc = None
        self.phase = 'CPU_VERIFIED'
        self.last_charge = 0.
        self.reserved = False
        self.abort = None
        self.arm_starts = {}
        self.watch_stop = threading.Event()

    def watchdog(self):
        while not self.watch_stop.wait(.25):
            now = time.monotonic()
            if held_seconds(self.leases, now) >= GPU_GUARD_SECONDS:
                self.abort = 'Aggregate GPU-time safety boundary reached'
            if self.started is not None and now-self.started >= GPU_GUARD_SECONDS:
                self.abort = 'Fixed cohort wall-time safety boundary reached'
            for proc,start in list(self.arm_starts.items()):
                if proc.poll() is None and now-start >= 1770:
                    self.abort = 'Per-arm thirty-minute safety boundary reached'
            if self.abort:
                for proc,_ in list(self.children):
                    if proc.poll() is None:
                        try:os.killpg(proc.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                time.sleep(2)
                for proc,_ in list(self.children):
                    if proc.poll() is None:
                        try:os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                return

    def status(self, phase, **extra):
        self.phase = phase
        record = dict(status=phase, host=os.uname().nodename, pid=os.getpid(),
                      observed_host_utc=datetime.now(timezone.utc).isoformat(),
                      aggregate_GPU_seconds=held_seconds(self.leases, time.monotonic()),
                      aggregate_GPU_hours_limit=2, maximum_parallel_arms=3,
                      active_lease_count=sum(x.get('ended') is None for x in self.leases),
                      source_commit='1a6d63db6baf7f6ad2a8e842a31341d2c128de25', **extra)
        write(ROOT / 'PARALLEL_STATUS.json', record)
        print(json.dumps(record), flush=True)

    def reserve(self):
        with self.budget.ledger() as data:
            assert not any(x['id'] == SESSION for x in data['gpu_sessions']), 'No automatic retry'
            committed = sum(x['reserved_gpu_hours'] if x['state'] in ('QUEUED', 'RUNNING') else
                            x.get('elapsed_seconds', 0) * x['ngpus'] / 3600 for x in data['gpu_sessions'])
            assert committed + 2 <= data['initial_total_gpu_hours'], 'Insufficient existing allowance'
            self.remaining_recorded = data['initial_total_gpu_hours'] - committed
            write(ROOT / 'PARALLEL_LEDGER_BEFORE_PRIVATE.json', data)
            data['gpu_sessions'].append(dict(id=SESSION, ngpus=1, reserved_gpu_hours=2.,
                elapsed_seconds=0., state='QUEUED', maximum_physical_devices=3,
                accounting='elapsed_seconds is summed physical-device-held seconds; ngpus1 is accounting multiplier',
                source_commit='1a6d63db6baf7f6ad2a8e842a31341d2c128de25',
                control_commit=self.binding['control_commit'], route='user_authorized_direct_ssh_pool_v2',
                pbs_allocation_claimed=False, user_parallel_override=True))
        self.reserved = True

    def charge(self, terminal=None):
        if not self.reserved:
            return
        used = held_seconds(self.leases, time.monotonic())
        with self.budget.ledger() as data:
            item = next(x for x in data['gpu_sessions'] if x['id'] == SESSION)
            item.update(elapsed_seconds=used, state=terminal or 'RUNNING', host=os.uname().nodename,
                        controller_pid=os.getpid(), active_physical_devices=sum(x.get('ended') is None for x in self.leases))
        self.last_charge = time.monotonic()
        write(ROOT / 'PARALLEL_GPU_ACCOUNTING.json', dict(aggregate_GPU_seconds=used,
            aggregate_GPU_hours=used / 3600, limit_GPU_hours=2, ledger=str(self.budget.LEDGER),
            state=terminal or 'RUNNING', leases=[{k:v for k,v in x.items() if k != 'receipt'} for x in self.leases]))

    def guard(self):
        now = time.monotonic()
        if self.abort:
            raise RuntimeError(self.abort)
        if held_seconds(self.leases, now) >= GPU_GUARD_SECONDS:
            raise RuntimeError('Aggregate two-GPU-hour safety deadline reached')
        if self.started is not None and now - self.started >= GPU_GUARD_SECONDS:
            raise RuntimeError('Fixed cohort two-hour wall window reached')
        if now - self.last_charge >= 20:
            self.charge()

    def claim(self, label):
        until = time.monotonic() + 60
        while time.monotonic() < until:
            self.guard()
            with self.pool.mutex():
                used = {x['card']['uuid'] for x in self.leases}
                options = [c for c in self.pool.candidates() if c['uuid'] not in used]
                if options:
                    card = options[0]
                    directory = self.pool.LOCKS / ('gpu' + str(card['index']))
                    if not directory.exists():
                        directory.mkdir()
                        self.pool.write(directory / 'pool_owner.json', dict(shared=True, v=2, pid=0, wid='pool_v2_shared'))
                    entry = directory / ('task_lhchannel_' + label + '.json')
                    assert not entry.exists(), 'Preserve any existing lease'
                    self.pool.write(entry, dict(name='lhchannel_' + label, wid='lhchannel_' + label,
                        pid=os.getpid(), pgid=None, cls='reader', need=16000, time=time.time()))
                    now = time.monotonic()
                    if self.started is None:
                        self.started = now
                        self.start_utc = datetime.now(timezone.utc)
                    receipt = ROOT / ('ALLOCATION_' + label + '.json')
                    self.leases.append(dict(card=card, entry=str(entry), started=now, ended=None, label=label, receipt=str(receipt)))
                    write(receipt, dict(status='VERIFIED', reservation_kind='authorized_pool_lease',
                        reservation_id=str(entry), ngpus=1, gpu_uuid=card['uuid'],
                        verified_by='Codex operator; explicit user pool and parallel authorization',
                        remaining_study_gpu_hours=self.remaining_recorded,
                        budget_ledger_reference=str(self.budget.LEDGER),
                        cohort_start_utc=self.start_utc.isoformat(),
                        cohort_deadline_utc=(self.start_utc + timedelta(hours=2)).isoformat(),
                        cohort_aggregate_GPU_hour_limit=2, maximum_concurrent_configuration_count=3,
                        pbs_allocation_claimed=False, exclusive_device_ownership_claimed=False,
                        control_commit=self.binding['control_commit'], user_override_file=str(ROOT / 'USER_PARALLEL_SCHEDULING_OVERRIDE.json')))
                    return self.leases[-1]
            time.sleep(3)
        raise RuntimeError('No compatible spare GPU lease; preserve active jobs')

    def launch(self, lease, args, log_name, script=SCRIPT):
        self.guard()
        assert sha(SCRIPT) == '95574295b76f16f5e5d8f70f01502848b4616fc7437030a5dfdf2ff1219f62f0'
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=lease['card']['uuid'], CES_GPU_ALLOCATION_RECEIPT=lease['receipt'],
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        bootstrap = ("import torch,os,sys,runpy,json;assert torch.cuda.is_available() and torch.cuda.device_count()==1;"
          "assert torch.cuda.is_bf16_supported() and 'H100' in torch.cuda.get_device_name(0);"
          "assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==os.environ['CUDA_VISIBLE_DEVICES'].removeprefix('GPU-');"
          "torch.cuda.set_per_process_memory_fraction(.18);print(json.dumps(dict(event='CUDA_BINDING',uuid=os.environ['CUDA_VISIBLE_DEVICES'],device=torch.cuda.get_device_name(0),torch=str(torch.__version__))),flush=True);"
          "sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')")
        log = (ROOT / log_name).open('x')
        argv = [PYTHON, '-u', '-c', bootstrap, str(script), *map(str, args)]
        log.write(json.dumps(dict(command=argv,lease=lease['entry'])) + '\n');log.flush()
        proc = subprocess.Popen(argv, cwd=SCIENCE, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with self.pool.mutex():
            data = self.pool.read(lease['entry']);data['pgid'] = proc.pid;self.pool.write(lease['entry'], data)
        self.children.append((proc, log))
        return proc

    def wait(self, proc, cap=1795):
        start = time.monotonic()
        while proc.poll() is None:
            self.guard()
            if time.monotonic() - start >= cap:
                raise RuntimeError('Stage or per-arm execution deadline reached')
            time.sleep(.5)
        if proc.returncode:
            raise RuntimeError('Stage failed; inspect preserved log')
        return time.monotonic() - start

    def run(self, lease, args, log, cap=1795, script=SCRIPT):
        return self.wait(self.launch(lease, args, log, script), cap)

    def cleanup(self):
        for proc, log in self.children:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL);proc.wait()
            log.close()
        for lease in self.leases:
            if lease.get('ended') is None:
                with self.pool.mutex():
                    p = Path(lease['entry'])
                    if p.exists() and self.pool.read(p).get('pid') == os.getpid():
                        if self.pool.alive(self.pool.read(p).get('pgid'), True):
                            raise RuntimeError('Own child process group still live; preserve lease')
                        p.unlink()
                lease['ended'] = time.monotonic()

    def execute(self):
        terminal = 'FAILED'
        try:
            self.reserve()
            watcher = threading.Thread(target=self.watchdog, daemon=True)
            watcher.start()
            primary = self.claim('primary')
            self.status('CUDA_SMOKE')
            self.run(primary, ['smoke', '--device', 'cuda', '--out', ROOT/'cuda_smoke'], 'cuda_smoke.log', cap=600)
            assert read(ROOT/'cuda_smoke/SMOKE_RESULT.json')['status'] == 'PASS'
            self.status('TRAIN_ONLY_MASK_AUDIT')
            self.run(primary, ['select-channels', '--config', ROOT/'audit.json', '--preflight', ROOT/'preflight',
                '--qualification', ROOT/'QUALIFIED.json', '--device', 'cuda', '--out', ROOT/'channel_audit'], 'channel_audit.log', cap=900)
            self.run(primary, ['bind-masks', '--config', ROOT/'audit.json', '--masks', ROOT/'channel_audit/MASKS.json', '--out', ROOT/'configs'], 'bind_masks.log', cap=120)
            dry = {}
            for arm in ARMS:
                self.status('REAL_INPUT_DRY_' + arm)
                wall = self.run(primary, ['dry-run', '--config', ROOT/'configs'/f'{arm}.json', '--preflight', ROOT/'preflight',
                    '--arm', 'D', '--device', 'cuda', '--out', ROOT/f'dry_{arm}'], f'dry_{arm}.log', cap=900)
                d = read(ROOT/f'dry_{arm}/DRY_RUN.json')
                assert d['status'] == 'PASS' and d['optimizer_steps_in_discarded_model'] == 1
                dry[arm] = dict(wall_seconds=wall, step_seconds=d['seconds_one_optimizer_step'])
            self.status('FROZEN_MASK_VALIDATION_RUNTIME_PROFILE')
            self.run(primary, ['--run', ROOT, '--out', ROOT/'RUNTIME_PROFILE.json'], 'runtime_profile.log', cap=900, script=CONTROL/'runtime_profile.py')
            profile = read(ROOT/'RUNTIME_PROFILE.json')
            steps = math.ceil(218266 / 32) * 3
            estimates = {arm:1.25*(dry[arm]['step_seconds']*steps + profile['setup_seconds'] +
                3*profile['TRAIN_real_calibration_forward_seconds'] + 4*profile['VAL_forward_seconds'] + 60) for arm in ARMS}
            used = held_seconds(self.leases, time.monotonic())
            passed = cost_gate(estimates, used)
            write(ROOT/'COST_GATE.json', dict(status='PASS' if passed else 'BLOCKED_RUNTIME_FIT',
                steps_per_arm=steps, estimates_seconds=estimates, actual_setup_GPU_seconds=used,
                dry_runs=dry, concurrency_safety_factor=1.25, per_arm_cap_seconds=1800,
                aggregate_GPU_seconds_limit=7200, masks_frozen_before_VAL_profile=True))
            if not passed:
                terminal = 'BLOCKED_RUNTIME_FIT';self.status(terminal, estimates_seconds=estimates);return
            extra = [self.claim('random'), self.claim('informed')]
            assignments = dict(zip(ARMS, [primary, *extra]))
            self.status('THREE_MATCHED_ARMS_TRAINING', assignments={a:l['card'] for a,l in assignments.items()})
            running = []
            for arm in ARMS:
                proc = self.launch(assignments[arm], ['train', '--config', ROOT/'configs'/f'{arm}.json',
                    '--preflight', ROOT/'preflight', '--arm', 'D', '--device', 'cuda',
                    '--qualification', ROOT/'QUALIFIED.json', '--dry-run-receipt', ROOT/f'dry_{arm}/DRY_RUN.json',
                    '--software-receipt', ROOT/'cuda_smoke/SMOKE_RESULT.json', '--out', ROOT/arm], f'train_{arm}.log')
                running.append((arm,proc,time.monotonic()))
                self.arm_starts[proc] = running[-1][2]
            while any(proc.poll() is None for _,proc,_ in running):
                self.guard()
                for arm,proc,start in running:
                    if proc.poll() is not None and proc.returncode:
                        raise RuntimeError('A matched arm failed: ' + arm)
                    if proc.poll() is None and time.monotonic()-start >= 1795:
                        raise RuntimeError('A matched arm exceeded its 30-minute bound: '+arm)
                time.sleep(.5)
            for arm,proc,_ in running:
                assert proc.returncode == 0
                result = read(ROOT/arm/'RUN_VERIFIED.json')
                assert result['status']=='COMPLETED_VERIFIED' and result['epochs_completed']==3
                assert not result['test_evaluated'] and sha(ROOT/arm/'best.pt')==result['selected_checkpoint_sha256']
            terminal = 'COMPLETED_VERIFIED'
            self.status(terminal)
        except Exception as error:
            write(ROOT/'PARALLEL_FAILURE.json',dict(error=str(error),phase=self.phase,no_automatic_retry=True))
            self.status('FAILED_OR_BLOCKED', error=str(error))
        finally:
            self.watch_stop.set()
            if 'watcher' in locals():watcher.join(timeout=3)
            try:
                self.cleanup()
            finally:
                self.charge(terminal)
        if terminal == 'COMPLETED_VERIFIED':
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            with (ROOT/'compare.log').open('x') as f:
                subprocess.run(['bash','CES_FL/proposals/lh_channel_selection_20261110_r1/commands.sh','compare',str(ROOT)],cwd=SCIENCE,env={**env,'CES_PYTHON':PYTHON},stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)
            self.status('COMPARISONS_COMPLETED_NO_EXTENSION')


if __name__ == '__main__':
    if '--verify-only' in sys.argv:
        print(json.dumps(dict(status='CPU_OPERATOR_VERIFY_PASS',binding=verify())),flush=True)
    else:
        Operator().execute()
