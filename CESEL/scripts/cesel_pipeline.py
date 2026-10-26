"""CESel seed-7 FF++ screen driver (decisions/2026-10-25-cesel-plan.md + Amendments 1-5). Several runners (one per PBS
job: gpu01 and gpu02) share ONE task list; a run is claimed through <out>.RUNLOCK (run/cesel_run.sh take_lock: owner
+ heartbeat on the shared filesystem), so a run is never executed twice; finished runs are skipped by every runner.

CESEL_DESIGN=p1 (default; Amendment 5, 26 Oct, "arms like B0"):
    B0 -> validate (cue JSON from B0 best, FF++ validation only) -> 8 arms p1_<arm> (sens inv floor full C_randcue
    C_shufgroup C_novalid B0_cfesD), each trained EXACTLY like B0: from CFES runs/clip_s<seed>_p1/last.pt, fresh
    heads / branches / CES state, 10 x 1976 steps, Amendment-9 selection with the start checkpoint as reference
    -> runs/ffpp_p1_<arm>_s<seed>;  each arm -> starve_p1_<arm> (CPU cue-starvation table, results/p1_cue_starvation)
    B0 -> starve_B0 (CPU)                       -> report (scripts/cesel_report.py --design p1, after everything)
CESEL_DESIGN=x5 (Amendment 1 warm-start screen, EXPLORATORY record only since Amendment 5):
    B0 -> validate -> 9 arms from B0 best (5 x 1976 steps) -> runs/ffpp_x5_<arm>_s<seed>
Each training task = train + evaluate_ces (full test sets, leak-free variants) + calibrate + posteval (A1 + per-cue).
Retry (Amendment 5): a failed run (rc != 0 or outputs missing; NOT exit 4 = held elsewhere) is retried up to MAX_TRIES
= 4 attempts IN TOTAL across all runners, with a randomised 10-20 min wait (RETRY_WAIT_MIN/MAX s) before the next
attempt; every attempt resumes from last.pt. Attempts are recorded in <out_root>/runs/attempts/<task>.jsonl (shared
by the runners: count, rc, host, job, earliest next start). STOP grace: CESEL_STOP_GRACE_S > 0 makes a runner that
finds work/STOP_CESEL during its first GRACE s wait (no launches, no exit) instead of exiting.
Scheduling: free GPU = allowed index (ALLOWED_GPUS) with >= 60 GB free AND < 10 % utilisation, not used by this
runner and not locked in work/gpulocks/<host>_<i>; at most MAX_CONC concurrent GPU runs per runner; re-checked every
POLL s. Other users' processes are never touched; a runner only waits on its own children.
A task held by another live runner (RUNLOCK heartbeat < 30 min old, or exit code 4) is skipped, not counted as a try.
STOP: the file work/STOP_CESEL stops new launches (running children finish; run/cesel_run.sh also refuses to start).
Exit: 0 when everything is done (the last runner writes the final report) or when every remaining task is held by
another live runner; 7 when nothing can run (a task used all its attempts).
Logs: logs/cesel_pipeline_<host>.log, one log per task in logs/cesel_runs/<task>.log.
Dry run (CPU, stub backbone, tiny budget): CESEL_DRY=1 CESEL_OUT=<scratch> python scripts/cesel_pipeline.py
"""
import json
import os
import random
import shlex
import socket
import subprocess
import sys
import time

A = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL"
PYD = "/home/sachin.chaudhary/.conda/envs/aiface/bin/python"
DRY = os.environ.get("CESEL_DRY", "0") == "1"
OUT = os.path.abspath(os.environ.get("CESEL_OUT", A))
DESIGN = os.environ.get("CESEL_DESIGN", "p1")
POLL = int(os.environ.get("POLL", "3" if DRY else "60"))
MAX_CONC = int(os.environ.get("MAX_CONC", "4"))
MAX_TRIES = int(os.environ.get("MAX_TRIES", "4"))                        # Amendment 5: up to 4 attempts per run
RETRY_WAIT = (float(os.environ.get("RETRY_WAIT_MIN", "2" if DRY else "600")),
              float(os.environ.get("RETRY_WAIT_MAX", "4" if DRY else "1200")))   # randomised 10-20 min
STOP_GRACE = float(os.environ.get("CESEL_STOP_GRACE_S", "0"))
DRY_FAIL = {x.split(":")[0]: int(x.split(":")[1]) for x in os.environ.get("CESEL_DRY_FAIL", "").split(",") if x}
T_START = time.time()
ALLOWED = [int(x) for x in os.environ.get("ALLOWED_GPUS", "0,1,2,3,4,6,7").split(",")]
MIN_FREE_MB, MAX_UTIL = 60 * 1024, 10
SEED = int(os.environ.get("SEED", "7"))
HOST = socket.gethostname().split(".")[0]
JOB = os.environ.get("PBS_JOBID", f"local{os.getpid()}")
LOCKS = os.path.join(A, "work", "gpulocks")
STOP = os.environ.get("CESEL_STOP_FILE", os.path.join(A, "work", "STOP_CESEL"))   # inherited by cesel_run.sh
LOG = os.path.join(OUT, "logs", f"cesel_pipeline_{HOST}.log")
RUNLOGS = os.path.join(OUT, "logs", "cesel_runs")
ARMS = ["B0", "full", "sens", "inv", "floor", "C_randcue", "C_shufgroup", "C_novalid", "B0_cfesD"]   # x5 launch order
P1_ARMS = ["full", "sens", "inv", "floor", "C_randcue", "C_shufgroup", "C_novalid", "B0_cfesD"]      # p1 launch order
ATTEMPTS = os.path.join(OUT, "runs", "attempts")


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{HOST} {JOB}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def ex(*p):
    return os.path.exists(os.path.join(OUT, *p))


def run_done(d):
    return lambda: all(ex(d, *p) for p in (("TRAINED",), ("eval", "summary_calibrated.csv"), ("eval", "a1.json"),
                                            ("eval", "interp.json")))


class Task:
    def __init__(self, name, args, done, deps=(), gpu=True, lockdir=None):
        self.name, self.args, self.done, self.deps, self.gpu = name, args, done, list(deps), gpu
        self.lockdir = lockdir                    # <out>.RUNLOCK of the task (None: CPU task, no cross-runner lock)
        self.proc, self.slot, self.failed, self.t0, self.tries = None, None, False, None, 0

    # ---- shared attempt record (Amendment 5 retry policy; one jsonl per task, appended by every runner)
    def _att_path(self):
        return os.path.join(ATTEMPTS, f"{DESIGN}_{self.name}.jsonl")

    def attempts(self):
        try:
            return [json.loads(l) for l in open(self._att_path()) if l.strip()]
        except (OSError, ValueError):
            return []

    def record(self, rec):
        os.makedirs(ATTEMPTS, exist_ok=True)
        with open(self._att_path(), "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def n_failed(self):
        return sum(1 for r in self.attempts() if not r.get("ok"))

    def exhausted(self):
        return not self.done() and self.n_failed() >= MAX_TRIES

    def waiting(self):
        """seconds left of the randomised retry wait after the last failed attempt (0 = may start)."""
        fails = [r for r in self.attempts() if not r.get("ok")]
        return max(0.0, float(fails[-1].get("next_ok", 0)) - time.time()) if fails else 0.0


def build_tasks():
    if DESIGN == "p1":
        return build_tasks_p1()
    if DESIGN != "x5":
        sys.exit(f"unknown CESEL_DESIGN {DESIGN}")
    b0 = f"runs/ffpp_B0_s{SEED}"
    T = [Task("B0", ["B0", str(SEED)], run_done(b0), lockdir=f"{b0}.RUNLOCK"),
         Task("validate", ["validate", str(SEED)], lambda: ex(b0, "validated_cues.json"), deps=["B0"],
              lockdir=f"{b0}.validate.RUNLOCK"),
         Task("starve_B0", ["starve", os.path.join(OUT, b0), "B0"],
              lambda: ex("results", "b0_cue_starvation", "B0_cue_starvation.done.json"), deps=["B0"], gpu=False,
              lockdir="results/b0_cue_starvation/B0.RUNLOCK")]
    for arm in ARMS:
        d = f"runs/ffpp_x5_{arm}_s{SEED}"
        T.append(Task(f"x5_{arm}", ["arm", arm, str(SEED)], run_done(d), deps=["validate"], lockdir=f"{d}.RUNLOCK"))
    return T


def build_tasks_p1():
    """Amendment 5: every arm trained like B0 (from clip_s<seed>_p1, 10 x 1976 steps); B0 itself is the reference."""
    b0 = f"runs/ffpp_B0_s{SEED}"
    T = [Task("B0", ["B0", str(SEED)], run_done(b0), lockdir=f"{b0}.RUNLOCK"),
         Task("validate", ["validate", str(SEED)], lambda: ex(b0, "validated_cues.json"), deps=["B0"],
              lockdir=f"{b0}.validate.RUNLOCK"),
         Task("starve_B0", ["starve", os.path.join(OUT, b0), "B0"],
              lambda: ex("results", "b0_cue_starvation", "B0_cue_starvation.done.json"), deps=["B0"], gpu=False,
              lockdir="results/b0_cue_starvation/B0.RUNLOCK")]
    for arm in P1_ARMS:
        d = f"runs/ffpp_p1_{arm}_s{SEED}"
        T.append(Task(f"p1_{arm}", ["arm_p1", arm, str(SEED)], run_done(d), deps=["validate"], lockdir=f"{d}.RUNLOCK"))
    for arm in P1_ARMS:
        d = f"runs/ffpp_p1_{arm}_s{SEED}"
        T.append(Task(f"starve_p1_{arm}", ["starve", os.path.join(OUT, d), f"p1_{arm}", "p1_cue_starvation"],
                      (lambda a=arm: ex("results", "p1_cue_starvation", f"p1_{a}_cue_starvation.done.json")),
                      deps=[f"p1_{arm}"], gpu=False, lockdir=f"results/p1_cue_starvation/p1_{arm}.RUNLOCK"))
    return T


# ------------------------------------------------------------------------------------------------ GPUs and locks
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
        if len(by) >= 3 and by[0] == "cesel_pipeline" and by[1] == HOST:
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
            and not os.path.exists(lock_dir(i))]


def acquire(i, name):
    if DRY:
        return True
    os.makedirs(LOCKS, exist_ok=True)
    try:
        os.mkdir(lock_dir(i))
    except FileExistsError:
        return False
    with open(os.path.join(lock_dir(i), "by"), "w") as f:
        f.write(f"cesel_pipeline {HOST} {os.getpid()} {JOB} {name}\n")
    return True


def release(i):
    if not DRY and i is not None:
        subprocess.run(["rm", "-rf", lock_dir(i)])


LOCK_LEASE = int(os.environ.get("CESEL_LOCK_LEASE", "1800"))     # = run/cesel_run.sh (Codex review F1 fix, 25 Oct)


def held_elsewhere(t):
    """True when another runner holds the task's RUNLOCK with a heartbeat younger than the lease (take_lock's clock
    rule; take_lock itself additionally observes the heartbeat counter before any takeover, so clock skew between
    hosts can make this a false 'not held' -> the launched runner exits 4 = HELD, never a double run)."""
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
    return age < LOCK_LEASE


# ------------------------------------------------------------------------------------------------ tasks
def launch(t, slot):
    env = dict(os.environ, CESEL_OUT=OUT)
    if DRY:
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4")
    elif t.gpu:
        env.update(CUDA_VISIBLE_DEVICES=str(slot), OMP_NUM_THREADS="2")
    else:
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    args = ["bash", os.path.join(A, "run", "cesel_run.sh")] + t.args
    n_prev = len(t.attempts())
    if DRY and DRY_FAIL.get(t.name, 0) > t.n_failed():          # dry-run test hook: simulated failed attempt
        args = ["bash", "-c", f"echo 'CESEL_DRY_FAIL: simulated failure of {t.name}'; exit 1"]
    lf = open(os.path.join(RUNLOGS, f"{t.name}.log"), "a")
    lf.write(f"\n{time.strftime('%Y-%m-%dT%H:%M:%S')} start on {HOST} job {JOB} slot {slot} (attempt record "
             f"{n_prev + 1}, failed so far {t.n_failed()}/{MAX_TRIES}): {shlex.join(args)}\n")
    lf.flush()
    t.proc, t.slot, t.t0 = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=A), slot, time.time()
    log(f"START {t.name} slot={slot} (failed attempts so far {t.n_failed()}/{MAX_TRIES}): {shlex.join(args[1:])}")


def report(final):
    args = [PYD, os.path.join(A, "scripts", "cesel_report.py"), "--out_root", OUT, "--seed", str(SEED),
            "--design", DESIGN] + ([] if final else ["--partial"])
    r = subprocess.run(args, capture_output=True, text=True)
    log(f"report ({'final' if final else 'partial'}): rc={r.returncode} {r.stdout[-1500:]}{r.stderr[-1500:]}")
    return r.returncode


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    os.makedirs(RUNLOGS, exist_ok=True)
    if DRY and OUT == A:
        sys.exit("CESEL_DRY=1 needs CESEL_OUT set to a scratch root")
    log(f"=== cesel_pipeline start out={OUT} dry={DRY} poll={POLL} max_conc={MAX_CONC} allowed={ALLOWED} seed={SEED} "
        f"MPS_PIPE={os.environ.get('CUDA_MPS_PIPE_DIRECTORY')}")
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
        for t in tasks:                                    # 1. reap
            if t.proc is not None and t.proc.poll() is not None:
                rc = t.proc.returncode
                if t.gpu:
                    release(t.slot)
                mins = (time.time() - t.t0) / 60
                if rc == 4:
                    log(f"HELD {t.name}: claimed by another runner ({mins:.1f} min); not counted as a try")
                else:
                    ok = rc == 0 and t.done()
                    t.tries += 1
                    rec = {"task": t.name, "design": DESIGN, "host": HOST, "job": JOB, "slot": t.slot, "rc": rc, "ok": ok,
                           "start": t.t0, "end": time.time(), "minutes": round(mins, 2),
                           "start_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t.t0))}
                    if not ok:
                        wait = random.uniform(*RETRY_WAIT)
                        rec["next_ok"] = time.time() + wait
                    t.record(rec)
                    nf = t.n_failed()
                    log(f"END {t.name} rc={rc} outputs={'ok' if t.done() else 'MISSING'} {mins:.1f} min "
                        f"(this runner's attempt {t.tries}; failed attempts in total {nf}/{MAX_TRIES})")
                    if not ok:
                        if nf < MAX_TRIES:
                            log(f"RETRY {t.name} after a randomised wait of {rec['next_ok'] - time.time():.0f} s "
                                f"(resumes from last.pt; see logs/cesel_runs/{t.name}.log)")
                        else:
                            t.failed = True
                            log(f"GIVE UP {t.name}: {nf} failed attempts (MAX_TRIES {MAX_TRIES})")
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
            if time.time() - T_START < STOP_GRACE:
                if not getattr(main, "_grace_logged", False):
                    log(f"STOP_CESEL present during the start grace ({STOP_GRACE:.0f} s): waiting for its removal")
                    main._grace_logged = True
                time.sleep(POLL)
                continue
            log("=== STOP_CESEL present: no new launches, nothing running -> exit")
            sys.exit(0)
        for t in tasks:                                    # attempts used up (by any runner)
            if not t.failed and t.proc is None and t.exhausted():
                t.failed = True
                log(f"GIVE UP {t.name}: {t.n_failed()} failed attempts recorded (MAX_TRIES {MAX_TRIES})")

        def eligible(t):
            return (not t.done() and not t.failed and t.proc is None and all(by[d].done() for d in t.deps)
                    and t.waiting() <= 0 and not held_elsewhere(t))
        cand = [t for t in tasks if eligible(t)]
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
            # otherwise: waiting for dependencies held by another runner
        if not stop:
            for t in [c for c in cand if not c.gpu]:       # CPU tasks: no GPU slot
                launch(t, None)
            gcand = [c for c in cand if c.gpu]
            n_gpu = sum(1 for t in tasks if t.proc is not None and t.gpu)
            if gcand and n_gpu < MAX_CONC:
                slots = free_slots({t.slot for t in tasks if t.proc is not None and t.gpu})
                for t, sl in zip(gcand, slots[:MAX_CONC - n_gpu]):
                    if acquire(sl, t.name):
                        launch(t, sl)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
