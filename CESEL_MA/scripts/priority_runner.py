"""CESEL_MA priority runner: ONE ordered task list (run/tasks.yaml, RE-READ every scheduling round, so the coordinator
can add / re-order / disable tasks without restarting) shared by several runners (one per PBS job: gpu01 and gpu02).
Same design as the fixed CESel runner (agents/CESEL/scripts/cesel_pipeline.py, Codex F1/F11 + Amendment 5):
  * a task is claimed through its RUNLOCK <MA_OUT>/work/tasklocks/<task>.RUNLOCK (run/ma_run.sh take_lock: token +
    atomic-rename compare-and-swap, heartbeat, fencing), so it never runs twice; exit 4 = held by another live runner;
  * free GPU = allowed index (ALLOWED_GPUS) with >= 60 GB free AND < 10 % utilisation, not used by this runner and not
    locked in CESEL_MA / CESEL / CLIPFAIR work/gpulocks/<host>_<i>; at most MAX_CONC concurrent GPU tasks per runner;
  * retry: <= MAX_TRIES (4) failed attempts per task IN TOTAL across runners (shared record runs/attempts/<task>.jsonl),
    randomised 10-20 min wait before the next attempt; every attempt resumes (the per-task scripts resume from last.pt);
  * STOP: work/STOP_MA stops new launches (running children finish); MA_STOP_GRACE_S > 0 = wait instead of exiting
    when STOP_MA is found during the first GRACE s;
  * HOLD: no launch while a file of HOLD_UNTIL_FILES is missing (the PBS output files of the CESel screen jobs, i.e.
    both screen jobs ended), at most HOLD_MAX_S (12 h); `touch work/HOLD_OFF` releases it at once.
Scheduling: every round, the READY tasks (enabled, not done, condition true, dependencies done, `needs` files present,
host allowed, retry wait over, not held elsewhere) are sorted by (priority, file order); CPU tasks start at once, GPU
tasks take free GPUs in that order -- a higher-priority ready task always launches first; tasks of the same priority
run concurrently up to the free GPUs / MAX_CONC.
Gate: every round while undecided, scripts/gate_check.py re-evaluates the step-1 gate from the CESel screen report
(validation only) -> results/gate.json. Conditions (task field `when`): always | gate_pass | gate_fail | gate_decided |
ma_not_worse | ma_worse (from results/ma_vs_single_s7.json); a condition that is decided FALSE skips the task.
Placeholders in run / done / needs / cmd / cwd: {MA} {CESEL} {CFES} {CLIPFAIR} {FAIRREL} {DING} {LABELS} {R} (roots) and
{ma_config} {single_axis_arm} (from gate.json; unresolved -> the task waits).
Exit: 0 when every task is done or skipped, or after IDLE_EXIT_S (default 2 h) with nothing running and nothing
launchable here (remaining tasks held by the other runner / waiting); 7 when only failed tasks remain.
Logs: logs/priority_<host>.log, logs/ma_runs/<task>.log; status: results/priority_status_<host>.json.
Dry run (CPU): MA_DRY=1 MA_OUT=<scratch> [MA_TASKS=<yaml>] [GATE_SCREEN_JSON=<json>] python scripts/priority_runner.py
"""
import json
import os
import random
import re
import shlex
import socket
import subprocess
import sys
import time

import yaml

A = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA"
AG = os.path.dirname(A)
R = os.path.dirname(AG)
PYD = "/home/sachin.chaudhary/.conda/envs/aiface/bin/python"
DRY = os.environ.get("MA_DRY", "0") == "1"
OUT = os.path.abspath(os.environ.get("MA_OUT", A))
TASKS_FILE = os.environ.get("MA_TASKS", os.path.join(A, "run", "tasks.yaml"))
POLL = int(os.environ.get("POLL", "2" if DRY else "60"))
MAX_CONC = int(os.environ.get("MAX_CONC", "4"))
MAX_TRIES = int(os.environ.get("MAX_TRIES", "4"))
RETRY_WAIT = (float(os.environ.get("RETRY_WAIT_MIN", "1" if DRY else "600")),
              float(os.environ.get("RETRY_WAIT_MAX", "2" if DRY else "1200")))
STOP_GRACE = float(os.environ.get("MA_STOP_GRACE_S", "0"))
IDLE_EXIT = float(os.environ.get("IDLE_EXIT_S", "20" if DRY else "7200"))
DRY_FAIL = {x.split(":")[0]: int(x.split(":")[1]) for x in os.environ.get("MA_DRY_FAIL", "").split(",") if x}
DRY_SLOTS = int(os.environ.get("MA_DRY_SLOTS", "2"))
RESERVE_MAX = float(os.environ.get("RESERVE_MAX_S", "3600"))
HOLD_FILES = [f for f in os.environ.get("HOLD_UNTIL_FILES", "").split(":") if f]
HOLD_MAX = float(os.environ.get("HOLD_MAX_S", "43200"))
HOLD_OFF = os.environ.get("HOLD_OFF_FILE", os.path.join(OUT if DRY else A, "work", "HOLD_OFF"))
T_START = time.time()
ALLOWED = [int(x) for x in os.environ.get("ALLOWED_GPUS", "0,1,2,3,4,6,7").split(",")]
MIN_FREE_MB, MAX_UTIL = 60 * 1024, 10
HOST = os.environ.get("MA_HOST_OVERRIDE") or socket.gethostname().split(".")[0]
JOB = os.environ.get("PBS_JOBID", f"local{os.getpid()}")
LOCKS = os.path.join(OUT if DRY else A, "work", "gpulocks")
OTHER_LOCKS = [] if DRY else [os.path.join(AG, "CESEL", "work", "gpulocks"), os.path.join(AG, "CLIPFAIR", "work", "gpulocks")]
STOP = os.environ.get("MA_STOP_FILE", os.path.join(OUT if DRY else A, "work", "STOP_MA"))
LOG = os.path.join(OUT, "logs", f"priority_{HOST}.log")
RUNLOGS = os.path.join(OUT, "logs", "ma_runs")
ATTEMPTS = os.path.join(OUT, "runs", "attempts")
LOCK_LEASE = int(os.environ.get("CESEL_LOCK_LEASE", "1800"))
ROOTS = {"MA": OUT, "R": R,
         "CESEL": os.environ.get("CESEL_ROOT", os.path.join(AG, "CESEL")),
         "CFES": os.environ.get("CFES_ROOT", os.path.join(AG, "CFES")),
         "CLIPFAIR": os.environ.get("CLIPFAIR_ROOT", os.path.join(AG, "CLIPFAIR")),
         "FAIRREL": os.environ.get("FAIRREL_ROOT", os.path.join(AG, "FAIRREL_REPRO")),
         "DING": os.environ.get("DING_ROOT", os.path.join(AG, "DINGREPRO_LIN")),
         "LABELS": os.environ.get("LABELS_ROOT", os.path.join(AG, "DEMO_AXES", "labels"))}
GATE_JSON = os.path.join(OUT, "results", "gate.json")
COMPARE_JSON = os.path.join(OUT, "results", "ma_vs_single_s7.json")
SCREEN_JSON = os.environ.get("GATE_SCREEN_JSON", os.path.join(ROOTS["CESEL"], "results", "cesel_ffpp_screen_p1.json"))
CONDITIONS = ("always", "gate_pass", "gate_fail", "gate_decided", "ma_not_worse", "ma_worse")
KINDS = ("ma", "ma_compare", "clipfair_sfd", "clipfair", "ding_gdagrad", "fairrel", "dingrepro", "armE", "ff2p1",
         "ff2_cfesd", "ff2_ces", "ff2_validate", "cmd", "cmd_cpu")


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{HOST} {JOB}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def jl(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ gate / conditions
class Facts:
    """gate.json / ma_vs_single_s7.json state, refreshed every round."""

    def __init__(self):
        self.gate, self.cmp, self._last_eval = None, None, 0.0

    def refresh(self):
        g = jl(GATE_JSON)
        if (g is None or g.get("status") not in ("PASS", "FAIL")) and time.time() - self._last_eval > (0 if DRY else 300):
            self._last_eval = time.time()
            args = [PYD, os.path.join(A, "scripts", "gate_check.py"), "--screen", SCREEN_JSON, "--out", GATE_JSON]
            if os.environ.get("GATE_RUNS_ROOT"):
                args += ["--runs_root", os.environ["GATE_RUNS_ROOT"]]
            r = subprocess.run(args, capture_output=True, text=True)
            log(f"gate_check rc={r.returncode}: {(r.stdout + r.stderr).strip()[-400:]}")
            g = jl(GATE_JSON)
        self.gate, self.cmp = g, jl(COMPARE_JSON)

    def status(self):
        return (self.gate or {}).get("status", "PENDING")

    def cond(self, c):
        """True / False (decided) / None (not yet known)."""
        st = self.status()
        if c == "always":
            return True
        if c in ("gate_pass", "gate_fail", "gate_decided"):
            if st not in ("PASS", "FAIL"):
                return None
            return {"gate_pass": st == "PASS", "gate_fail": st == "FAIL", "gate_decided": True}[c]
        if c in ("ma_not_worse", "ma_worse"):
            if st == "FAIL":
                return False
            if st != "PASS" or self.cmp is None:
                return None
            nw = bool(self.cmp.get("not_worse"))
            return nw if c == "ma_not_worse" else not nw
        raise ValueError(c)

    def placeholders(self):
        d = dict(ROOTS)
        g = self.gate or {}
        if g.get("status") == "PASS":
            d["ma_config"], d["single_axis_arm"] = g.get("ma_config"), g.get("single_axis_arm")
        return d


_PH = re.compile(r"\{([A-Za-z_]+)\}")


def subst(s, ph):
    """-> string or None when a placeholder is unknown."""
    miss = [m for m in _PH.findall(str(s)) if ph.get(m) is None]
    if miss:
        return None
    return _PH.sub(lambda m: str(ph[m.group(1)]), str(s))


# ------------------------------------------------------------------------------------------------ tasks
class Task:
    def __init__(self, spec, idx):
        self.spec, self.idx = spec, idx
        self.name = spec["name"]
        self.proc, self.slot, self.t0, self.tries, self.failed = None, None, None, 0, False

    def update(self, spec, idx):
        self.spec, self.idx = spec, idx

    @property
    def prio(self):
        return float(self.spec.get("priority", 100))

    @property
    def gpu(self):
        return bool(self.spec.get("gpu", True))

    def enabled(self):
        return bool(self.spec.get("enabled", True))

    def resolved(self, key, ph):
        v = self.spec.get(key)
        if v is None:
            return [] if key in ("done", "needs", "deps", "run") else None
        if isinstance(v, list):
            out = [subst(x, ph) for x in v]
            return None if any(x is None for x in out) else out
        return subst(v, ph)

    def done(self, ph):
        d = self.resolved("done", ph)
        return bool(d) and all(os.path.exists(p) for p in d)

    # ---- shared attempt record
    def _att(self):
        return os.path.join(ATTEMPTS, f"{self.name}.jsonl")

    def attempts(self):
        try:
            return [json.loads(l) for l in open(self._att()) if l.strip()]
        except (OSError, ValueError):
            return []

    def record(self, rec):
        os.makedirs(ATTEMPTS, exist_ok=True)
        with open(self._att(), "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def n_failed(self):
        return sum(1 for r in self.attempts() if not r.get("ok"))

    def waiting(self):
        f = [r for r in self.attempts() if not r.get("ok")]
        return max(0.0, float(f[-1].get("next_ok", 0)) - time.time()) if f else 0.0

    def lockdir(self):
        return os.path.join(OUT, "work", "tasklocks", f"{self.name}.RUNLOCK")


def validate_spec(t):
    if not isinstance(t, dict) or not isinstance(t.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", t["name"]):
        return "task needs a name of [A-Za-z0-9_.-]"
    if not isinstance(t.get("run"), list) or not t["run"] or t["run"][0] not in KINDS:
        return f"{t.get('name')}: run must be a list starting with a kind in {KINDS}"
    if not t.get("done"):
        return f"{t['name']}: done (marker files) required"
    if t.get("when", "always") not in CONDITIONS:
        return f"{t['name']}: when must be one of {CONDITIONS}"
    if t["run"][0] in ("cmd", "cmd_cpu") and not t.get("cmd"):
        return f"{t['name']}: kind cmd needs a cmd field"
    return None


_INC_CACHE = {}          # include path -> (mtime, normalised task list) of the last version that parsed and validated


def normalise_external(t, inc):
    """External task (include file) -> internal spec. External schema (README "External task files"):
    {name, cmd (str or list), done [...], deps [...] (names in the same file), needs [...], gpu (default true), cwd,
    env {...}, when, enabled, reserve, hosts}; or an internal `run` list. priority = the include's priority (file order
    breaks ties); names / deps get the include's prefix."""
    if not isinstance(t, dict) or "name" not in t:
        return t
    pre = inc.get("prefix", "")
    o = dict(t)
    o["name"] = pre + str(t["name"])
    o["deps"] = [pre + str(d) for d in (t.get("deps") or [])]
    if "run" not in o and o.get("cmd") is not None:
        o["run"] = ["cmd" if o.get("gpu", True) else "cmd_cpu"]
    if isinstance(o.get("cmd"), list):
        o["cmd"] = shlex.join(str(x) for x in o["cmd"])
    o["priority"] = float(inc.get("priority", 50))
    if "reserve" not in o and inc.get("reserve") is not None:
        o["reserve"] = bool(inc["reserve"])
    o["source"] = inc["_path"]
    return o


def read_includes(spec):
    """-> (list of normalised external tasks, signature). A missing optional file contributes no tasks; a file that
    fails to parse or validate keeps its previous good version (logged)."""
    out, sig = [], []
    for inc in spec.get("include") or []:
        path = subst(inc.get("file", ""), ROOTS)
        if not path:
            log(f"include without a resolvable file: {inc}")
            continue
        try:
            mt = os.path.getmtime(path)
        except OSError:
            mt = None
        sig.append((path, mt))
        if mt is None:
            if not inc.get("optional", True):
                log(f"required include {path} missing")
            _INC_CACHE.pop(path, None)
            continue
        cached = _INC_CACHE.get(path)
        if cached and cached[0] == mt:
            out += cached[1]
            continue
        try:
            raw = yaml.safe_load(open(path)) or {}
            lst = raw.get("tasks") if isinstance(raw, dict) else raw
            norm = [normalise_external(t, dict(inc, _path=path)) for t in (lst or [])]
            errs = [e for e in (validate_spec(t) for t in norm) if e]
            if errs:
                raise ValueError(errs[:5])
            _INC_CACHE[path] = (mt, norm)
            log(f"include {path}: {len(norm)} tasks (priority {inc.get('priority', 50)})")
            out += norm
        except Exception as e:      # noqa: BLE001
            log(f"INCLUDE ERROR {path}: {e!r}; using its previous version ({len(cached[1]) if cached else 0} tasks)")
            if cached:
                out += cached[1]
    return out, tuple(sig)


def load_tasks(tasks, last_ok):
    """Re-read the task file (+ its include files); keep the previous list on any error. Returns (tasks, signature)."""
    try:
        mt = os.path.getmtime(TASKS_FILE)
        spec = yaml.safe_load(open(TASKS_FILE)) or {}
        inc_tasks, inc_sig = read_includes(spec)
        sig = (mt, inc_sig)
        if sig == last_ok:
            return tasks, last_ok
        lst = list(spec.get("tasks") or []) + inc_tasks
        errs = [e for e in (validate_spec(t) for t in lst) if e]
        names = [t["name"] for t in lst if isinstance(t, dict) and "name" in t]
        if len(set(names)) != len(names):
            errs.append(f"duplicate task names {sorted({n for n in names if names.count(n) > 1})}")
        known = set(names)
        errs += [f"{t['name']}: unknown dependency {d}" for t in lst for d in (t.get("deps") or []) if d not in known]
        if errs:
            log(f"TASK FILE ERROR ({TASKS_FILE} + includes), keeping the previous list: {errs[:5]}")
            return tasks, sig if tasks else last_ok
        new = {}
        for i, t in enumerate(lst):
            if t["name"] in tasks:
                tasks[t["name"]].update(t, i)
                new[t["name"]] = tasks[t["name"]]
            else:
                new[t["name"]] = Task(t, i)
        for n, t in tasks.items():          # removed from the file while running: keep it until it ends
            if n not in new and t.proc is not None:
                new[n] = t
                log(f"task {n} removed from the file while running here; it finishes, no new launch")
                t.spec = dict(t.spec, enabled=False)
        log(f"task list (re)loaded: {len(lst)} tasks ({len(inc_tasks)} from includes) from {TASKS_FILE}")
        return new, sig
    except Exception as e:      # noqa: BLE001
        log(f"TASK FILE ERROR ({TASKS_FILE}): {e!r}; keeping the previous list")
        return tasks, last_ok


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


def locked_anywhere(i):
    return os.path.exists(lock_dir(i)) or any(os.path.exists(os.path.join(d, f"{HOST}_{i}")) for d in OTHER_LOCKS)


def clear_own_stale_locks():
    if not os.path.isdir(LOCKS):
        return
    for d in os.listdir(LOCKS):
        p = os.path.join(LOCKS, d, "by")
        if not os.path.exists(p):
            continue
        by = open(p).read().split()
        if len(by) >= 3 and by[0] == "priority_runner" and by[1] == HOST:
            try:
                os.kill(int(by[2]), 0)
                alive = True
            except (OSError, ValueError):
                alive = False
            if not alive:
                subprocess.run(["rm", "-rf", os.path.join(LOCKS, d)])
                log(f"removed stale GPU lock of a dead runner: {d} ({' '.join(by)})")


def free_slots(used):
    if DRY:
        return [i for i in range(DRY_SLOTS) if i not in used and not os.path.exists(lock_dir(i))]
    try:
        st = gpu_state()
    except Exception as e:      # noqa: BLE001
        log(f"nvidia-smi failed: {e}")
        return []
    return [i for i in ALLOWED if i in st and i not in used and st[i][0] >= MIN_FREE_MB and st[i][1] < MAX_UTIL
            and not locked_anywhere(i)]


def acquire(i, name):
    os.makedirs(LOCKS, exist_ok=True)
    try:
        os.mkdir(lock_dir(i))
    except FileExistsError:
        return False
    with open(os.path.join(lock_dir(i), "by"), "w") as f:
        f.write(f"priority_runner {HOST} {os.getpid()} {JOB} {name}\n")
    return True


def release(i):
    if i is not None:
        subprocess.run(["rm", "-rf", lock_dir(i)])


def held_elsewhere(t):
    L = t.lockdir()
    if not os.path.isdir(L):
        return False
    hb = os.path.join(L, "heartbeat")
    try:
        age = time.time() - os.path.getmtime(hb if os.path.exists(hb) else L)
    except OSError:
        return False
    return age < LOCK_LEASE


# ------------------------------------------------------------------------------------------------ launch
def launch(t, slot, ph):
    run = t.resolved("run", ph)
    env = dict(os.environ, MA_OUT=OUT, TASK_NAME=t.name, TASK_GPU="1" if t.gpu else "0",
               TASK_DONE=":".join(t.resolved("done", ph) or []), **{f"{k}_ROOT": v for k, v in ROOTS.items()
                                                                     if k not in ("MA", "R", "LABELS")})
    if t.spec.get("cmd"):
        env["TASK_CMD"] = subst(t.spec["cmd"], ph)
        env["TASK_CWD"] = subst(t.spec.get("cwd", A), ph)
    for k, v in (t.spec.get("env") or {}).items():
        env[str(k)] = subst(str(v), ph)
    if DRY:
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    elif t.gpu:
        env.update(CUDA_VISIBLE_DEVICES=str(slot), OMP_NUM_THREADS="2")
    else:
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    args = ["bash", os.path.join(A, "run", "ma_run.sh")] + run
    if DRY and DRY_FAIL.get(t.name, 0) > t.n_failed():
        args = ["bash", "-c", f"echo 'MA_DRY_FAIL: simulated failure of {t.name}'; exit 1"]
    lf = open(os.path.join(RUNLOGS, f"{t.name}.log"), "a")
    lf.write(f"\n{time.strftime('%Y-%m-%dT%H:%M:%S')} start on {HOST} job {JOB} slot {slot} priority {t.prio} "
             f"(failed attempts so far {t.n_failed()}/{MAX_TRIES}): {shlex.join(args)}\n")
    lf.flush()
    t.proc, t.slot, t.t0 = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=A), slot, time.time()
    log(f"START {t.name} (P{t.prio:g}) slot={slot}: {shlex.join(args[2:])}")


def write_status(tasks, facts, ph, state):
    rows = []
    for t in sorted(tasks.values(), key=lambda t: (t.prio, t.idx)):
        rows.append({"name": t.name, "priority": t.prio, "state": state.get(t.name), "slot": t.slot,
                     "failed_attempts": t.n_failed()})
    p = os.path.join(OUT, "results", f"priority_status_{HOST}.json")
    tmp = p + f".tmp{os.getpid()}"
    json.dump({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": HOST, "job": JOB, "gate": facts.status(),
               "ma_not_worse": (facts.cmp or {}).get("not_worse"), "tasks": rows}, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


def main():
    for d in (os.path.dirname(LOG), RUNLOGS, os.path.join(OUT, "results"), os.path.join(OUT, "work", "tasklocks")):
        os.makedirs(d, exist_ok=True)
    if DRY and OUT == A:
        sys.exit("MA_DRY=1 needs MA_OUT set to a scratch root")
    log(f"=== priority_runner start out={OUT} tasks={TASKS_FILE} dry={DRY} poll={POLL} max_conc={MAX_CONC} "
        f"allowed={ALLOWED} MPS_PIPE={os.environ.get('CUDA_MPS_PIPE_DIRECTORY')} screen={SCREEN_JSON}")
    if not DRY and HOST not in ("gpu01", "gpu02"):
        sys.exit(f"must run on gpu01 / gpu02 inside PBS, not {HOST}")
    if not DRY:
        log(f"gpu state {gpu_state()}")
    clear_own_stale_locks()
    facts, tasks, mt = Facts(), {}, None
    idle_since, grace_logged, last_state = None, False, None
    while True:
        tasks, mt = load_tasks(tasks, mt)
        facts.refresh()
        ph = facts.placeholders()
        # 1. reap
        for t in tasks.values():
            if t.proc is not None and t.proc.poll() is not None:
                rc = t.proc.returncode
                if t.gpu:
                    release(t.slot)
                mins = (time.time() - t.t0) / 60
                if rc == 4:
                    log(f"HELD {t.name}: claimed by another runner ({mins:.1f} min); not counted as a try")
                else:
                    ok = rc == 0 and t.done(ph)
                    t.tries += 1
                    rec = {"task": t.name, "host": HOST, "job": JOB, "slot": t.slot, "rc": rc, "ok": ok, "start": t.t0,
                           "end": time.time(), "minutes": round(mins, 2)}
                    if not ok:
                        rec["next_ok"] = time.time() + random.uniform(*RETRY_WAIT)
                    t.record(rec)
                    nf = t.n_failed()
                    log(f"END {t.name} rc={rc} outputs={'ok' if t.done(ph) else 'MISSING'} {mins:.1f} min "
                        f"(failed attempts in total {nf}/{MAX_TRIES})")
                    if not ok and nf >= MAX_TRIES:
                        t.failed = True
                        log(f"GIVE UP {t.name}: {nf} failed attempts")
                    elif not ok:
                        log(f"RETRY {t.name} after {rec['next_ok'] - time.time():.0f} s (resumes; logs/ma_runs/{t.name}.log)")
                t.proc, t.slot = None, None
        facts.refresh()
        ph = facts.placeholders()
        # 2. classify (memoised over dependencies: a dependency that is skipped / failed / disabled skips its dependants)
        state, ready = {}, []

        def classify(t, stack=()):
            if t.name in state:
                return state[t.name]
            if t.name in stack:
                state[t.name] = "skipped"
                log(f"dependency cycle at {t.name}: skipped")
                return "skipped"
            st = _classify(t, stack + (t.name,))
            state[t.name] = st
            return st

        def _classify(t, stack):
            if t.proc is not None:
                return "running"
            if t.done(ph):
                return "done"
            if not t.failed and t.n_failed() >= MAX_TRIES:
                t.failed = True
                log(f"GIVE UP {t.name}: {t.n_failed()} failed attempts recorded")
            if t.failed:
                return "failed"
            if not t.enabled():
                return "disabled"
            c = facts.cond(t.spec.get("when", "always"))
            if c is False:
                return "skipped"
            ds = [classify(tasks[d], stack) for d in (t.spec.get("deps") or []) if d in tasks]
            if any(x in ("skipped", "failed", "disabled") for x in ds):
                return "skipped"                      # a dependency will never be done
            if c is True and ds and all(x in ("done", "running", "ready", "held_elsewhere") for x in ds) and \
                    any(x != "done" for x in ds):
                return "waiting_dep"                  # only waits for dependencies that run now / launch first
            if c is None or any(x != "done" for x in ds):
                return "waiting"
            needs, run = t.resolved("needs", ph), t.resolved("run", ph)
            if needs is None or run is None or not all(os.path.exists(p) for p in needs):
                return "waiting"
            hosts = t.spec.get("hosts")
            if hosts and HOST not in hosts:
                return "other_host"
            if t.waiting() > 0:
                return "retry_wait"
            if held_elsewhere(t):
                return "held_elsewhere"
            return "ready"
        for t in tasks.values():
            if classify(t) == "ready":
                ready.append(t)
        ready.sort(key=lambda t: (t.prio, t.idx))
        running = [t for t in tasks.values() if t.proc is not None]
        summary = {s: sum(1 for v in state.values() if v == s) for s in sorted(set(state.values()))}
        if summary != last_state:
            log(f"state {summary}; gate {facts.status()}; ready {[t.name for t in ready][:8]}")
            last_state = summary
        write_status(tasks, facts, ph, state)
        # 3. exit conditions
        if tasks and all(v in ("done", "skipped", "disabled") for v in state.values()):
            log("=== every task done or skipped -> exit 0")
            sys.exit(0)
        stop = os.path.exists(STOP)
        if stop and not running:
            if time.time() - T_START < STOP_GRACE:
                if not grace_logged:
                    log(f"STOP_MA present during the start grace ({STOP_GRACE:.0f} s): waiting for its removal")
                    grace_logged = True
                time.sleep(POLL)
                continue
            log("=== STOP_MA present: no new launches, nothing running -> exit 0")
            sys.exit(0)
        if not running and not ready and not any(v == "waiting_dep" for v in state.values()):
            if tasks and all(v in ("done", "skipped", "disabled", "failed") for v in state.values()):
                log(f"=== nothing can run: failed {[n for n, v in state.items() if v == 'failed']} -> exit 7")
                sys.exit(7)
            idle_since = idle_since or time.time()
            if time.time() - idle_since > IDLE_EXIT:
                log(f"=== idle {IDLE_EXIT:.0f} s with nothing launchable here (states {summary}) -> exit 0")
                sys.exit(0)
        else:
            idle_since = None
        # 4. launch (priority order; CPU tasks need no slot). Reservation: a `reserve` GPU task that only waits for a
        # running dependency (e.g. BD arms behind bd_smoke) keeps one free GPU per such task away from LOWER-priority
        # tasks, for at most RESERVE_MAX_S. Hold: no launch at all while a HOLD_UNTIL_FILES file is missing.
        now = time.time()
        for t in tasks.values():
            if state.get(t.name) == "waiting_dep" and t.spec.get("reserve") and t.gpu:
                t.reserve_since = getattr(t, "reserve_since", None) or now
            else:
                t.reserve_since = None
        reservers = [t for t in tasks.values() if t.reserve_since and now - t.reserve_since < RESERVE_MAX]
        hold = [f for f in HOLD_FILES if not os.path.exists(f)] if now - T_START < HOLD_MAX else []
        if hold and os.path.exists(HOLD_OFF):           # coordinator override: release the hold without a restart
            hold = []
        if hold and ready:
            if not getattr(main, "_hold_logged", False):
                log(f"HOLD: no launches until these exist (screen jobs ended): {hold}")
                main._hold_logged = True
        elif getattr(main, "_hold_logged", False) and not hold:
            log("HOLD released")
            main._hold_logged = False
        if not stop and not hold:
            n_gpu = sum(1 for t in running if t.gpu)
            used = {t.slot for t in running if t.gpu}
            slots = None
            for t in ready:
                if not t.gpu:
                    launch(t, None, ph)
                    continue
                if n_gpu >= MAX_CONC:
                    continue
                if slots is None:
                    slots = free_slots(used)
                n_res = sum(1 for r in reservers if r.prio < t.prio)
                if len(slots) <= n_res:
                    continue
                while len(slots) > n_res:
                    sl = slots.pop(0)
                    if acquire(sl, t.name):
                        launch(t, sl, ph)
                        used.add(sl)
                        n_gpu += 1
                        break
        time.sleep(POLL)


if __name__ == "__main__":
    main()
