"""Amendment 5 checks (26 Oct): (1) report --design p1 compares the arms with B0 itself (no B0-continued), applies the
same keep / drop / winner rules, fails closed while an arm is pending, writes results/cesel_ffpp_screen_p1.*; the x5
report is marked exploratory. (2) driver retry policy: shared attempt record, at most MAX_TRIES failed attempts, the
randomised wait blocks a restart, a successful attempt ends it. (3) p1 task graph: 8 arms + 8 starvation tasks, arm deps.
  CUDA_VISIBLE_DEVICES="" ~/.conda/envs/pulid/bin/python tests/test_amend5_p1.py"""
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import cesel_report as R  # noqa: E402

S = ROOT / "tests" / "_amend5_scratch"
shutil.rmtree(S, ignore_errors=True)
fx, cf = S / "fx", S / "cfes"


def trained(p, v):
    p.mkdir(parents=True, exist_ok=True)
    (p / "TRAINED").write_text(json.dumps({"best": v, "best_epoch": 3, "epoch": 10}))


def setup(vals, b0=90.0):
    shutil.rmtree(fx, ignore_errors=True)
    shutil.rmtree(cf, ignore_errors=True)
    trained(fx / "runs/ffpp_B0_s7", b0)
    trained(fx / "runs/ffpp_x5_B0_s7", 50.0)          # a B0-continued run must be IGNORED by the p1 design
    for arm, v in vals.items():
        trained(fx / f"runs/ffpp_p1_{arm}_s7", v)
    for seed, v in zip((7, 17, 27), (87.62, 89.68, 89.28)):
        trained(cf / f"runs_rev4/clip_D_w0.3_s{seed}", v)


def run(*extra):
    R.CF = str(cf)
    sys.argv = ["cesel_report.py", "--out_root", str(fx), "--design", "p1", *extra]
    R.main()
    return json.loads((fx / "results/cesel_ffpp_screen_p1.json").read_text())


base = {"sens": 89, "inv": 91, "floor": 92, "full": 95, "C_randcue": 80, "C_shufgroup": 80, "C_novalid": 80,
        "B0_cfesD": 90}
setup(base)
j = run()
d = j["decisions"]
assert j["complete"] and j["design"] == "p1" and not j["exploratory"]
assert d["reference_run"] == "B0" and d["keep"]["sens"] is False and d["keep"]["inv"] is True, d
assert d["winner"] == "floor" and d["promote"] == ["B0", "B0+CFES-D", "+floor"], d
md = (fx / "results/cesel_ffpp_screen_p1.md").read_text()
assert "vs B0:" in md and "B0-continued" not in md.split("## Test metrics")[0], "p1 report must not use B0-continued"
assert "Per-arm cue starvation" in md
print("PASS p1a: arms compared with B0 (90), not the x5 B0-continued run (50); +sens fails -> Full excluded; winner +floor")
setup(dict(base, sens=93, C_novalid=94.5))
d = run()["decisions"]
assert d["dropped_terms"] == ["inv"] and d["winner"] == "sens" and not d["unresolved"], d
print("PASS p1b: C_novalid matches Full within 2.06 -> inv dropped; winner +sens")
setup({k: v for k, v in base.items() if k != "C_randcue"})
j = run("--partial")
assert not j["complete"] and j["decisions"]["promote"] is None and "drop:C_randcue" in j["decisions"]["unresolved"]
print("PASS p1c: pending control -> fail closed, report partial")
shutil.rmtree(fx / "runs/ffpp_B0_s7")
j = run("--partial", "--force")
assert j["decisions"]["promote"] is None and not j["complete"]
print("PASS p1d: B0 missing -> every keep gate unresolved, no promotion")
# x5 report is marked exploratory and keeps its own file
setup(base)
for arm, v in dict(base, B0=90).items():
    trained(fx / f"runs/ffpp_x5_{arm}_s7", v)
sys.argv = ["cesel_report.py", "--out_root", str(fx)]
R.main()
jx = json.loads((fx / "results/cesel_ffpp_screen.json").read_text())
assert jx["exploratory"] and jx["design"] == "x5" and "EXPLORATORY" in (fx / "results/cesel_ffpp_screen.md").read_text()
print("PASS x5: warm-start report marked EXPLORATORY, separate file")

# ---- driver: retry policy + p1 graph (no launches)
os.environ.update(CESEL_DRY="1", CESEL_OUT=str(S / "drv"), CESEL_DESIGN="p1", MAX_TRIES="4", RETRY_WAIT_MIN="600",
                  RETRY_WAIT_MAX="1200")
import cesel_pipeline as P  # noqa: E402
P = importlib.reload(P)
T = P.build_tasks()
names = [t.name for t in T]
assert names[:3] == ["B0", "validate", "starve_B0"] and len(T) == 3 + 8 + 8, names
arm = {t.name: t for t in T}
assert all(arm[f"p1_{a}"].deps == ["validate"] and arm[f"p1_{a}"].args == ["arm_p1", a, "7"] for a in P.P1_ARMS)
assert all(arm[f"starve_p1_{a}"].deps == [f"p1_{a}"] and not arm[f"starve_p1_{a}"].gpu for a in P.P1_ARMS)
print("PASS graph: B0, validate, starve_B0 + 8 arm_p1 tasks (dep validate) + 8 CPU starvation tasks (dep own arm)")
t = arm["p1_full"]
for i in range(3):
    t.record({"task": t.name, "ok": False, "rc": 1, "next_ok": time.time() + 900})
assert t.n_failed() == 3 and not t.exhausted() and 600 < t.waiting() <= 900
t2 = P.Task("p1_full", t.args, t.done)               # a second runner sees the same record
assert t2.n_failed() == 3 and t2.waiting() > 600
t.record({"task": t.name, "ok": False, "rc": 1, "next_ok": time.time() - 1})
assert t.exhausted() and t2.exhausted() and t.waiting() == 0
print("PASS retry: attempts shared across runners; wait blocks restart; 4 failed attempts -> exhausted")
w = [P.random.uniform(*P.RETRY_WAIT) for _ in range(1000)]
assert 600 <= min(w) and max(w) <= 1200 and P.MAX_TRIES == 4
print("PASS retry wait: uniform in [600, 1200] s")
shutil.rmtree(S, ignore_errors=True)
print("ALL_PASS test_amend5_p1")
