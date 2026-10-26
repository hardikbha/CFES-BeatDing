"""CLIPFAIR driver (Amendment 3): 4 baselines x 3 seeds (7, 17, 27) on FF++ (Lin), the CESel runner pattern
(scripts/cesel_pipeline.py of agents/CESEL, read only). Several runners (one per PBS job: gpu01 and gpu02) share ONE
task list; a run is claimed through <out>.RUNLOCK (run/clipfair_run.sh take_lock: owner + heartbeat on the shared
filesystem), so a run is never executed twice; finished runs are skipped by every runner.

Tasks (dependencies):  sfd_s<seed> (Ding SFD, GPU, ~minutes) -> ding_s<seed>
                       daw_s<seed>, dag_s<seed>, pgfdd_s<seed> (independent)              -> report (after everything)
Each training task = train + evaluate_clipfair (full test sets, leak-free variants) + calibrate + A1 posteval.
Scheduling: free GPU = allowed index (ALLOWED_GPUS) with >= 60 GB free AND < 10 % utilisation, not used by this runner
and not locked in CLIPFAIR/work/gpulocks/<host>_<i> nor in CESEL/work/gpulocks/<host>_<i> (CESel's locks are only read);
at most MAX_CONC concurrent GPU runs per runner; re-checked every POLL s. Other users' processes are never touched.
STOP: the file work/STOP_CLIPFAIR stops new launches (running children finish; run/clipfair_run.sh also refuses).
Exit: 0 when everything is done (the last runner writes the final report) or when every remaining task is held by
another live runner; 7 when nothing can run (failures); a failed run is retried once (MAX_TRIES 2) per runner.
Logs: logs/clipfair_pipeline_<host>.log, one log per task in logs/clipfair_runs/<task>.log.
Dry run (CPU, stub backbone, tiny budget): CLIPFAIR_DRY=1 CLIPFAIR_OUT=<scratch> python scripts/clipfair_pipeline.py
"""
import os
import shlex
import socket
import subprocess
import sys
import time

A = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CLIPFAIR"
CESEL_LOCKS = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL/work/gpulocks"
PYD = "/home/sachin.chaudhary/.conda/envs/aiface/bin/python"
DRY = os.environ.get("CLIPFAIR_DRY", "0") == "1"
OUT = os.path.abspath(os.environ.get("CLIPFAIR_OUT", A))
POLL = int(os.environ.get("POLL", "3" if DRY else "60"))
MAX_CONC = int(os.environ.get("MAX_CONC", "4"))
MAX_TRIES = int(os.environ.get("MAX_TRIES", "2"))
ALLOWED = [int(x) for x in os.environ.get("ALLOWED_GPUS", "0,1,2,3,4,6,7").split(",")]
MIN_FREE_MB, MAX_UTIL = 60 * 1024, 10
SEEDS = [int(s) for s in os.environ.get("SEEDS", "7,17,27").split(",")]
METHODS = [m for m in os.environ.get("METHODS", "pgfdd,daw,dag,ding").split(",")]   # launch order (PG-FDD longest)
HOST = socket.gethostname().split(".")[0]
JOB = os.environ.get("PBS_JOBID", f"local{os.getpid()}")
LOCKS = os.path.join(A, "work", "gpulocks")
STOP = os.path.join(A, "work", "STOP_CLIPFAIR")
LOG = os.path.join(OUT, "logs", f"clipfair_pipeline_{HOST}.log")
RUNLOGS = os.path.join(OUT, "logs", "clipfair_runs")


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{HOST} {JOB}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def ex(*p):
    return os.path.exists(os.path.join(OUT, *p))


def run_done(d):
    return lambda: all(ex(d, *p) for p in (("TRAINED",), ("eval", "summary_calibrated.csv"), ("eval", "a1.json")))


class Task:
    def __init__(self, name, args, done, deps=(), lockdir=None):
        self.name, self.args, self.done, self.deps, self.gpu = name, args, done, list(deps), True
        self.lockdir = lockdir
        self.proc, self.slot, self.failed, self.t0, self.tries = None, None, False, None, 0


def build_tasks():
    T = []
    for s in SEEDS:
        if "ding" in METHODS:
            d = f"runs/ffpp_ding_sfd_s{s}"
            T.append(Task(f"sfd_s{s}", ["sfd", str(s)], (lambda d=d: ex(d, "sfd_channels.json")), lockdir=f"{d}.RUNLOCK"))
        for m in METHODS:
            d = f"runs/ffpp_{m}_s{s}"
            T.append(Task(f"{m}_s{s}", ["train", m, str(s)], run_done(d), deps=[f"sfd_s{s}"] if m == "ding" else [],
                          lockdir=f"{d}.RUNLOCK"))
    return T


def gpu_state():
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60)
    out = {}
    for line in r.stdout.strip().splitlines():
        i, mf, ut = [x.strip() for x in line.split(",")]
        out[int(i)] = (float(mf), float(ut))
    return out


def lock_dir(i):
    return os.path.join(LOCKS, f"{HOST}_{i}")


def clear_own_stale_locks():
    if DRY or not os.path.isdir(LOCKS):
        return
    for d in os.listdir(LOCKS):
        p = os.path.join(LOCKS, d, "by")
        if not os.path.exists(p):
            continue
        by = open(p).read().split()
        if len(by) >= 3 and by[0] == "clipfair_pipeline" and by[1] == HOST:
            try:
                os.kill(int(by[2]), 0)
                alive = True
            except (OSError, ValueError):
                alive = False
            if not alive:
                subprocess.run(["rm", "-rf", os.path.join(LOCKS, d)])
                log(f"removed stale GPU lock of a dead driver: {d} ({' '.join(by)})")


def free_slots(used):
    if DRY:
        return [f"cpu{i}" for i in range(2) if f"cpu{i}" not in used]
    try:
        st = gpu_state()
    except Exception as e:      # noqa: BLE001
        log(f"nvidia-smi failed: {e}")
        return []
    return [i for i in ALLOWED if i in st and i not in used and st[i][0] >= MIN_FREE_MB and st[i][1] < MAX_UTIL
            and not os.path.exists(lock_dir(i)) and not os.path.exists(os.path.join(CESEL_LOCKS, f"{HOST}_{i}"))]


def acquire(i, name):
    if DRY:
        return True
    os.makedirs(LOCKS, exist_ok=True)
    try:
        os.mkdir(lock_dir(i))
    except FileExistsError:
        return False
    with open(os.path.join(lock_dir(i), "by"), "w") as f:
        f.write(f"clipfair_pipeline {HOST} {os.getpid()} {JOB} {name}\n")
    return True


def release(i):
    if not DRY and i is not None:
        subprocess.run(["rm", "-rf", lock_dir(i)])


def held_elsewhere(t):
    if not t.lockdir:
        return False
    L = os.path.join(OUT, t.lockdir)
    if not os.path.isdir(L):
        return False
    hb = os.path.join(L, "heartbeat")
    try:
        age = time.time() - os.path.getmtime(hb if os.path.exists(hb) else L)
    except OSError:
        return False
    return age < int(os.environ.get("CESEL_LOCK_LEASE", "1800"))   # = run/clipfair_run.sh RUNLOCK lease (Codex F1 fix)


def launch(t, slot):
    env = dict(os.environ, CLIPFAIR_OUT=OUT)
    if DRY:
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    else:
        env.update(CUDA_VISIBLE_DEVICES=str(slot), OMP_NUM_THREADS="2")
    args = ["bash", os.path.join(A, "run", "clipfair_run.sh")] + t.args
    lf = open(os.path.join(RUNLOGS, f"{t.name}.log"), "a")
    lf.write(f"\n{time.strftime('%Y-%m-%dT%H:%M:%S')} start on {HOST} job {JOB} slot {slot}: {shlex.join(args)}\n")
    lf.flush()
    t.proc, t.slot, t.t0 = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=A), slot, time.time()
    log(f"START {t.name} slot={slot}: {shlex.join(args[1:])}")


def report(final):
    args = [PYD, os.path.join(A, "scripts", "clipfair_report.py"), "--out_root", OUT,
            "--seeds", ",".join(map(str, SEEDS))] + ([] if final else ["--partial"])
    r = subprocess.run(args, capture_output=True, text=True)
    log(f"report ({'final' if final else 'partial'}): rc={r.returncode} {r.stdout[-800:]}{r.stderr[-1500:]}")
    return r.returncode


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    os.makedirs(RUNLOGS, exist_ok=True)
    if DRY and OUT == A:
        sys.exit("CLIPFAIR_DRY=1 needs CLIPFAIR_OUT set to a scratch root")
    log(f"=== clipfair_pipeline start out={OUT} dry={DRY} poll={POLL} max_conc={MAX_CONC} allowed={ALLOWED} "
        f"seeds={SEEDS} methods={METHODS} MPS_PIPE={os.environ.get('CUDA_MPS_PIPE_DIRECTORY')}")
    if not DRY and HOST not in ("gpu01", "gpu02"):
        sys.exit(f"must run on gpu01 / gpu02 inside PBS, not {HOST}")
    if not DRY:
        log(f"PBS CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} (each run gets its own index); "
            f"gpu state {gpu_state()}")
    clear_own_stale_locks()
    tasks = build_tasks()
    by = {t.name: t for t in tasks}
    last_partial = 0.0
    while True:
        for t in tasks:
            if t.proc is not None and t.proc.poll() is not None:
                rc = t.proc.returncode
                release(t.slot)
                mins = (time.time() - t.t0) / 60
                if rc == 4:
                    log(f"HELD {t.name}: claimed by another runner ({mins:.1f} min); not counted as a try")
                else:
                    t.tries += 1
                    log(f"END {t.name} rc={rc} outputs={'ok' if t.done() else 'MISSING'} {mins:.1f} min (attempt {t.tries})")
                    if not (rc == 0 and t.done()):
                        if t.tries < MAX_TRIES:
                            log(f"RETRY {t.name} later (resumes from last.pt; see logs/clipfair_runs/{t.name}.log)")
                        else:
                            t.failed = True
                t.proc, t.slot = None, None
                if time.time() - last_partial > 60:
                    report(final=False)
                    last_partial = time.time()
        running = [t for t in tasks if t.proc is not None]
        if all(t.done() for t in tasks):
            rc = report(final=True)
            log(f"=== all tasks done; final report rc={rc}")
            sys.exit(0 if rc == 0 else 6)
        stop = os.path.exists(STOP)
        if stop and not running:
            log("=== STOP_CLIPFAIR present: no new launches, nothing running -> exit")
            sys.exit(0)

        def eligible(t):
            return (not t.done() and not t.failed and t.proc is None and all(by[d].done() for d in t.deps)
                    and not held_elsewhere(t))
        cand = [t for t in tasks if eligible(t)]
        cand.sort(key=lambda t: 0 if t.name.startswith("sfd") else 1)       # SFD first (short; unblocks Ding)
        if not cand and not running:
            pending = [t for t in tasks if not t.done()]
            failed = [t.name for t in tasks if t.failed]
            if failed:
                log(f"=== nothing can run here: failed {failed}; pending {[t.name for t in pending]}")
                report(final=False)
                sys.exit(7)
            if pending and all(held_elsewhere(t) for t in pending):
                log(f"=== every remaining task is held by another live runner: {[t.name for t in pending]} -> exit")
                sys.exit(0)
        if not stop and cand:
            n_gpu = len(running)
            if n_gpu < MAX_CONC:
                slots = free_slots({t.slot for t in running})
                for t, sl in zip(cand, slots[:MAX_CONC - n_gpu]):
                    if acquire(sl, t.name):
                        launch(t, sl)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
